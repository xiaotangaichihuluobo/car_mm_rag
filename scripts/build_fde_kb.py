# -*- coding: utf-8 -*-
"""
MUVERA 阶段1 建库: 读已缓存的页 patch 向量(page_vecs/p*.npy)压成 FDE 定长向量入库。
  秒级、纯 numpy、不占显存、不出网(不 import torch)。改 FDE 参数后用 --rebuild 重扫。

【R1】不出网不加载模型。【R2b】集合新建 car_mm.colqwen2_page_fde, 既有对象不动。

跑法:
    PYTHONPATH=. python scripts/build_fde_kb.py [--rebuild]
    --rebuild   删集合重头建(改了任一 fde_* 参数后必须带)
"""

import argparse
import os
import sys
import time

import store

# Windows 控制台默认 GBK, 中文日志会乱码 -> 强制 UTF-8(踩过坑, 别删)
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
from rag_qa.core.fde_encoder import FDEEncoder
from rag_qa.core.image_store import MultimodalFDEStore
from _milvus_util import ensure_indexed

conf = Config()

# 与 build_multimodal_kb.py 共用同一份向量缓存(它写的, 我们只读)。
# 多书后缓存按书分目录: page_vecs/{source_id}/, 这里按书读。
VEC_CACHE_ROOT = os.path.join(project_root, 'rag_qa', 'data', 'page_vecs')


def vec_cache_dir(source_id):
    """某书的向量缓存目录: page_vecs/{source_id}/。

    :param source_id: 书号
    :return: 绝对路径
    """
    return os.path.join(VEC_CACHE_ROOT, source_id)


def section(title):
    """打印阶段分隔线 + 标题。"""
    print(f'\n{"=" * 64}\n{title}\n{"=" * 64}', flush=True)


# ---------------------------------------------------------------- ① 读缓存
def load_page_vecs(source_id):
    """读某书全部已缓存的页向量; 缺缓存就明确报错, 不静默跳过。

    :param source_id: 书号(决定读哪个缓存子目录)
    :return: {页号: np.ndarray}
    :raises RuntimeError: 缓存目录不存在或为空
    """
    cache_dir = vec_cache_dir(source_id)
    section(f'① 读页向量缓存: {cache_dir}')
    if not os.path.isdir(cache_dir):
        raise RuntimeError(f'向量缓存目录不存在: {cache_dir}\n'
                           f'先跑 scripts/build_multimodal_kb.py --source {source_id} 建多模态库。')

    files = sorted(f for f in os.listdir(cache_dir) if f.endswith('.npy'))
    if not files:
        raise RuntimeError(f'向量缓存是空的: {cache_dir}\n'
                           f'先跑 scripts/build_multimodal_kb.py --source {source_id}。')

    page_vecs = {}
    for f in files:
        page_vecs[int(f[1:4])] = np.load(os.path.join(cache_dir, f))
    print(f'① 读到 {len(page_vecs)} 页向量(每页 ~{max(v.shape[0] for v in page_vecs.values())} patch)',
          flush=True)
    return page_vecs


# ---------------------------------------------------------------- ② 生成 FDE
def generate_fde(page_vecs):
    """页 patch 向量 -> FDE 定长向量(纯 numpy, 不加载模型、不占显存、不出网)。

    :param page_vecs: {页号: patch 向量}
    :return: {页号: FDE 定长向量}
    """
    section('② 生成 FDE(纯 numpy, 不重跑 ColQwen2)')
    encoder = FDEEncoder()
    print(f'② FDE 参数: k_sim={encoder.k_sim} dim_proj={encoder.dim_proj} '
          f'reps={encoder.r_reps} seed={encoder.seed} -> dim={encoder.dim}', flush=True)

    t = time.time()
    page_fdes = {}
    for i, (page, vecs) in enumerate(sorted(page_vecs.items())):
        page_fdes[page] = encoder.encode_document(vecs)
        if (i + 1) % 100 == 0 or i + 1 == len(page_vecs):
            print(f'  已生成 {i + 1}/{len(page_vecs)} 页, 用时 {time.time() - t:.0f}s', flush=True)
    print(f'② 完成: {len(page_fdes)} 页, 用时 {time.time() - t:.0f}s', flush=True)
    return page_fdes


# ---------------------------------------------------------------- ③ 入库
def store(page_fdes, source_id, rebuild=False):
    """写入 Milvus; 该书已建且页数吻合则跳过(但 --rebuild 时整集合重建)。

    集合是**共享**的, "已建好跳过"按书统计 —— 不能拿"集合总页数 == 本书页数"判。

    :param page_fdes: {页号: FDE 向量}
    :param source_id: 本书号
    :param rebuild: 删掉集合重头建(会连别的书一起清, 只在正式重建时用)
    :return: MultimodalFDEStore(已加载)
    """
    section(f'③ 写入 Milvus({conf.MM_FDE_COLLECTION_NAME} @ {conf.MM_MILVUS_DATABASE})')
    store = MultimodalFDEStore()
    encoder = FDEEncoder()

    if not rebuild and store.has_collection() and _source_fde_count(store, source_id) == len(page_fdes):
        print(f'③ 该书已有 {len(page_fdes)} 页且页数吻合, 跳过入库', flush=True)
        # 注意: 跳过入库**不会**自愈坏序 —— 只重跑 ensure_indexed。
        #   真乱序时要 --rebuild(或 compact), "重跑一遍"和"compact"是两件事。
        ensure_indexed(store, len(page_fdes), timeout=120)
        store.load()
        return store

    store.create_collection(dim=encoder.dim, drop_existing=True)
    t = time.time()
    n = store.insert_pages(page_fdes, source_id, batch_size=64)
    print(f'③ 入库 {n} 页, 用时 {time.time() - t:.0f}s', flush=True)
    ensure_indexed(store, len(page_fdes), timeout=120)
    store.load()
    print(f'③ 集合现有 {store.count()} 页(已加载)', flush=True)
    return store


def _source_fde_count(store, source_id):
    """某书在共享 FDE 集合里的行数; 集合不存在 / 没书号时返回 0。

    :param store: MultimodalFDEStore
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
    ap = argparse.ArgumentParser(description='MUVERA 阶段1 的 FDE 建库')
    ap.add_argument('--source', default=None, help='书号(读该书缓存, 并作为 source_id 入库)')
    ap.add_argument('--rebuild', action='store_true',
                    help='删掉 FDE 集合重头建(改了任一 fde_* 参数后必须带这个)')
    args = ap.parse_args()

    source_id = args.source or conf.MM_DEFAULT_BOOK
    t0 = time.time()
    print(f'source_id  : {source_id}')
    print(f'向量缓存   : {vec_cache_dir(source_id)}')
    print(f'Milvus     : {conf.MILVUS_HOST}:{conf.MILVUS_PORT} / '
          f'库={conf.MM_MILVUS_DATABASE} / 集合={conf.MM_FDE_COLLECTION_NAME}')

    page_vecs = load_page_vecs(source_id)
    page_fdes = generate_fde(page_vecs)
    # 注意别写成 `store = store(...)`: 赋值会让 `store` 变成 main 的局部变量,
    #   从而遮蔽模块级的同名函数 store()——在 RHS 调用时就 UnboundLocalError 了。
    store(page_fdes, source_id, rebuild=args.rebuild)

    print(f'\n{"=" * 64}')
    print(f'FDE 建库完成: {len(page_fdes)} 页, 总用时 {time.time() - t0:.0f}s')
    print(f'{"=" * 64}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
