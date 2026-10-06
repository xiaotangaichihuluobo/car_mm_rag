# -*- coding: utf-8 -*-
"""
删一本书: 从多模态页库(页向量 + 页 FDE)删掉该书 + 清其磁盘缓存。也是"改书内容"
  的前半段(删掉后再重跑 build_multimodal_kb.py / build_fde_kb.py 重新入库)。只动这
  本书——靠 source_id 软删 + 子目录。
  Milvus delete 是软删墓碑, 删完 ensure_indexed 刷新索引覆盖状态。

跑法:
    PYTHONPATH=. python scripts/delete_book_kb.py train_a
"""

import argparse
import os
import shutil
import sys

# Windows 控制台默认 GBK, 中文日志会乱码 -> 强制 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 路径配置: 本文件在 scripts/, 向上一级即项目根(base/ 所在层)。
#   同时把 scripts/ 目录本身加进 sys.path, 以便 import 同级的 _milvus_util。
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
for p in (project_root, current_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from base.config import Config
from rag_qa.core.image_store import MultimodalPageStore, MultimodalFDEStore
from _milvus_util import ensure_indexed

conf = Config()

VEC_CACHE_ROOT = os.path.join(project_root, 'rag_qa', 'data', 'page_vecs')


def _rmtree(d):
    """删目录(不存在时静默)。"""
    if os.path.isdir(d):
        shutil.rmtree(d)
        return True
    return False


def delete_book(source_id, dry_run=False):
    """从三条腿的共享集合删掉某书, 并清该书磁盘缓存。

    :param source_id: 书号
    :param dry_run: 只打印会做什么, 不动任何东西
    :return: 0 成功(或 dry_run 打印完); 非 0 出错
    """
    print(f'\n{"=" * 64}\n删除书: {source_id}\n{"=" * 64}', flush=True)
    if source_id not in conf.MM_BOOKS:
        print(f'[错误] 书号 {source_id!r} 不在书目注册表 {list(conf.MM_BOOKS)} 里,'
              f'无从删起。', flush=True)
        return 1

    # 每个 store 都有 delete_by_source; 共享集合里其它书的行不受影响
    destinations = [
        ('页向量', MultimodalPageStore()),
        ('页 FDE ', MultimodalFDEStore()),
    ]
    for label, store in destinations:
        if dry_run:
            print(f'  [dry-run] 软删 {label}: source_id == "{source_id}"', flush=True)
            continue
        n = store.delete_by_source(source_id)
        print(f'  {label}: 软删 {n} 行', flush=True)

    # 磁盘缓存按书分目录, 直接整目录删
    for label, d in (('页图', conf.book_image_dir(source_id)),
                     ('页向量', os.path.join(VEC_CACHE_ROOT, source_id))):
        if dry_run:
            print(f'  [dry-run] 删 {label} 目录: {d}', flush=True)
            continue
        print(f'  {label} 目录 {"已删" if _rmtree(d) else "不存在(跳过)"}: {d}', flush=True)

    if dry_run:
        print('\n[dry-run] 结束, 未做任何更改', flush=True)
        return 0

    # 软删后让索引覆盖新状态 —— 不刷新的话, 墓碑可能有残留行被检索到
    for label, store in destinations:
        if store.has_collection():
            ensure_indexed(store, store.count(), timeout=120)
            print(f'  {label}: 索引覆盖校验完成(现有 {store.count()} 行)', flush=True)

    print(f'\n删除完成: {source_id} 的多模态页库与缓存已清净。\n'
          f'若只是改这本书的内容, 现在重跑:\n'
          f'    PYTHONPATH=. python scripts/build_multimodal_kb.py --source {source_id}\n'
          f'    PYTHONPATH=. python scripts/build_fde_kb.py --source {source_id}', flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description='删一本书的三条腿 + 磁盘缓存')
    ap.add_argument('source_id', help='要删的书号(必须在 config.ini [books] 注册表里)')
    ap.add_argument('--dry-run', action='store_true', help='只打印会做什么, 不动任何东西')
    args = ap.parse_args()
    return delete_book(args.source_id, dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(main())