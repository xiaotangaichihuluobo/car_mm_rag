# -*- coding: utf-8 -*-
"""
L4 多模态建库: PDF 每页 -> 页图 -> ColQwen2 多向量 -> Milvus。三个阶段都断点续跑。

【R1】编码全程本地 ColQwen2, 不碰 API key。【R2b】专用库 car_mm(新建, 既有对象不动)。

跑法:
    PYTHONPATH=. python scripts/build_multimodal_kb.py [--pdf <路径>] [--rebuild] [--skip-render] [--limit N]
    --rebuild   删集合重头建(向量缓存也清空)
    --limit N   只处理前 N 页(调试用, 别建正式库)
"""

import argparse
import os
import sys
import time

# Windows 控制台默认 GBK, 中文日志会乱码 -> 强制 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 路径配置: 本文件在 scripts/, 向上一级即项目根(base/ 所在层)。
#   同时把 scripts/ 目录本身加进 sys.path, 以便 import 同级的 _milvus_util。
current_dir = os.path.dirname(os.path.abspath(__file__))     # scripts
project_root = os.path.dirname(current_dir)                  # 项目根
for p in (project_root, current_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from base.config import Config
from rag_qa.core.colqwen2_encoder import ColQwen2Encoder
from rag_qa.core.pdf_render import render_pdf_pages
from rag_qa.core.image_store import MultimodalPageStore
from _milvus_util import ensure_indexed

conf = Config()

# 页向量缓存根: 一页一个 .npy, 便于断点续跑(整包 npz 每次重写代价太大)。
# 多书后**必须按书分目录** —— 不同书同页号会互相覆盖, 跨书污染是静默的坏库。
VEC_CACHE_ROOT = os.path.join(project_root, 'rag_qa', 'data', 'page_vecs')


def vec_cache_dir(source_id):
    """某书的向量缓存目录: page_vecs/{source_id}/。目录不存在时懒创建。

    :param source_id: 书号
    :return: 绝对路径
    """
    d = os.path.join(VEC_CACHE_ROOT, source_id)
    os.makedirs(d, exist_ok=True)
    return d


def vec_cache_path(source_id, page):
    """某书某页的向量缓存路径: page_vecs/{source_id}/pNNN.npy。

    :param source_id: 书号
    :param page: 页号
    :return: .npy 绝对路径
    """
    return os.path.join(vec_cache_dir(source_id), f'p{int(page):03d}.npy')


def section(title):
    """打印阶段分隔线 + 标题; 各阶段开头都用它, 避免每个函数各写一遍分隔线模板。"""
    print(f'\n{"=" * 64}\n{title}\n{"=" * 64}', flush=True)


# ---------------------------------------------------------------- ① 渲染
def render(pdf_path, source_id):
    """PDF 每页渲染成 PNG(落到该书的页图子目录)。

    :param pdf_path: PDF 路径
    :param source_id: 书号(决定页图子目录)
    :return: 页清单 [{'page': int, 'png_path': str}, ...]
    :raises FileNotFoundError: PDF 不存在
    """
    out_dir = conf.book_image_dir(source_id)
    section(f'① 渲染页图: {pdf_path} -> {out_dir}')
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f'PDF 不存在: {pdf_path}')
    t = time.time()
    pages = render_pdf_pages(pdf_path, out_dir)
    print(f'① 渲染完成: {len(pages)} 页, 用时 {time.time() - t:.0f}s '
          f'-> {out_dir}', flush=True)
    return pages


# ---------------------------------------------------------------- ② 编码
def encode(pages, source_id):
    """页图 -> ColQwen2 多向量, 逐页落盘 page_vecs/{source_id}/。

    已编码的页跳过(断点续跑); 全程本地, 不调 API。

    :param pages: render 返回的页清单; 缺缓存的页会被编码
    :param source_id: 书号(决定向量缓存子目录)
    """
    section('② 本地 ColQwen2 编码(不调 API)')
    os.makedirs(vec_cache_dir(source_id), exist_ok=True)

    # 先筛出"还没编码过"的页 —— 续跑时这一批通常很小
    todo = [p for p in pages if not os.path.exists(vec_cache_path(source_id, p['page']))]
    print(f'② 待编码 {len(todo)}/{len(pages)} 页(其余命中向量缓存)', flush=True)
    if not todo:
        return

    from PIL import Image

    encoder = ColQwen2Encoder()
    t = time.time()
    for i, item in enumerate(todo):
        page = item['page']
        img = Image.open(item['png_path']).convert('RGB')
        vecs = encoder.encode_images([img])[0]
        # 逐页落盘: 一旦中途崩, 已编码的页不会白跑
        np.save(vec_cache_path(source_id, page), vecs)
        del img
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            done = len(pages) - len(todo) + i + 1
            print(f'  编码 {done}/{len(pages)} 页, 每页 {vecs.shape[0]} patch, '
                  f'用时 {time.time() - t:.0f}s', flush=True)


# ---------------------------------------------------------------- ③ 入库
def load_vectors(pages, source_id):
    """把全部页向量载入内存(354 页 × 747 × 128 × 4B ≈ 135MB, 无压力); 缺缓存的页报错。

    :param pages: render 返回的页清单
    :param source_id: 书号(决定读哪个缓存子目录)
    :return: {页号: np.ndarray(patches, 128)}
    :raises RuntimeError: 有页缺向量缓存
    """
    section(f'③ 写入 Milvus({conf.MM_COLLECTION_NAME} @ {conf.MM_MILVUS_DATABASE})')
    page_vecs = {}
    missing = []
    for item in pages:
        page = item['page']
        fp = vec_cache_path(source_id, page)
        if os.path.exists(fp):
            page_vecs[page] = np.load(fp)
        else:
            missing.append(page)
    if missing:
        raise RuntimeError(f'这些页没有向量缓存, 先跑阶段②: {missing[:10]}...')
    print(f'③ 载入 {len(page_vecs)} 页向量', flush=True)
    return page_vecs


def compute_max_capacity(page_vecs):
    """patches 数组容量上限; 必须 >= 全库单页最多 patch 数, 否则 insert 报错。

    :param page_vecs: load_vectors 返回的 {页号: 向量}
    :return: 容量上限 int(单页最多 patch × 1.2 + 16 留余量)
    """
    max_len = max(v.shape[0] for v in page_vecs.values())
    cap = int(max_len * 1.2) + 16
    print(f'③ 单页最多 {max_len} patch -> max_capacity={cap}', flush=True)
    return cap


def store_pages(page_vecs, source_id, rebuild):
    """入库; 该书已建好且页数吻合则跳过(但 --rebuild 时整集合重建)。

    集合是**共享**的(所有书共用一个页面集合), 所以"已建好"按书统计, 不能拿
    "集合总页数 == 本书页数"判 —— 那是旧单书语义, 多书下恒不成立。

    :param page_vecs: {页号: 向量}
    :param source_id: 本书号
    :param rebuild: 整集合重建(会连别的书一起清 —— 只在正式重建时用)
    :return: MultimodalPageStore(已加载)
    """
    store = MultimodalPageStore()

    if not rebuild and store.has_collection() and _source_page_count(store, source_id) == len(page_vecs):
        print(f'③ 该书已有 {len(page_vecs)} 页且页数吻合, 跳过入库', flush=True)
        ensure_indexed(store, len(page_vecs))
        store.load()
        return store

    store.create_collection(max_capacity=compute_max_capacity(page_vecs), drop_existing=True)
    t = time.time()
    n = store.insert_pages(page_vecs, source_id, batch_size=16)
    print(f'③ 入库 {n} 页, 用时 {time.time() - t:.0f}s', flush=True)
    ensure_indexed(store, len(page_vecs))
    store.load()
    print(f'③ 集合现有 {store.count()} 页(已加载)', flush=True)
    return store


def _source_page_count(store, source_id):
    """某书在共享集合里的页行数; 集合不存在 / 没书号时返回 0。

    :param store: MultimodalPageStore
    :param source_id: 书号
    :return: int
    """
    if not source_id or not store.has_collection():
        return 0
    rows = store.client.query(collection_name=store.collection_name,
                              filter=f'source_id == "{source_id}"',
                              output_fields=['id'])
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description='L4 多模态知识库全量建库')
    ap.add_argument('--source', default=None, help='书号; 给了就用该书对应的 PDF, 并作为 source_id 入库')
    ap.add_argument('--pdf', default=None, help='PDF 路径(默认取 config.ini, 或 --source 对应的书)')
    ap.add_argument('--rebuild', action='store_true', help='删掉集合与向量缓存, 重头建(整集合, 含其它书)')
    ap.add_argument('--skip-render', action='store_true', help='跳过渲染阶段')
    ap.add_argument('--limit', type=int, default=None, help='只处理前 N 页(调试用)')
    args = ap.parse_args()

    # --source 优先: 书号 -> 该书 PDF; 否则退回 config.ini 的默认书
    source_id = args.source or conf.MM_DEFAULT_BOOK
    pdf_path = conf.book_pdf(source_id) if args.source else (args.pdf or conf.MM_PDF_PATH)
    image_dir = conf.book_image_dir(source_id)
    vec_dir = vec_cache_dir(source_id)

    t0 = time.time()
    print(f'source_id  : {source_id}')
    print(f'PDF        : {pdf_path}')
    print(f'页图目录   : {image_dir}')
    print(f'向量缓存   : {vec_dir}')
    print(f'Milvus     : {conf.MILVUS_HOST}:{conf.MILVUS_PORT} / '
          f'库={conf.MM_MILVUS_DATABASE} / 集合={conf.MM_COLLECTION_NAME}')

    if args.rebuild:
        print('\n--rebuild: 清空**全部书的**向量缓存与页面集合, 全部重头建', flush=True)
        if os.path.isdir(VEC_CACHE_ROOT):
            for root, _dirs, files in os.walk(VEC_CACHE_ROOT, topdown=False):
                for f in files:
                    if f.endswith('.npy'):
                        os.remove(os.path.join(root, f))
        MultimodalPageStore().create_collection(max_capacity=64, drop_existing=True)

    # ① 页清单: 渲染成新 PNG, 或 --skip-render 时从该书已有 PNG 反推
    if args.skip_render:
        import glob
        files = sorted(glob.glob(os.path.join(image_dir, 'p*.png')))
        pages = [{'page': int(os.path.basename(f)[1:4]), 'png_path': f} for f in files]
        print(f'\n① 跳过渲染, 从 {image_dir} 读到 {len(pages)} 页', flush=True)
    else:
        pages = render(pdf_path, source_id)

    # --limit 是调试开关: 页清单统一在这里裁剪一次, 各阶段不再各切一遍
    if args.limit:
        pages = pages[:args.limit]

    # ② 编码
    encode(pages, source_id)

    # ③ 入库
    store_pages(load_vectors(pages, source_id), source_id, rebuild=args.rebuild)

    print(f'\n{"=" * 64}')
    print(f'建库完成: {len(pages)} 页, 总用时 {time.time() - t0:.0f}s')
    print(f'{"=" * 64}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
