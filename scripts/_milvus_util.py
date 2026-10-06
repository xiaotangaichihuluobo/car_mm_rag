# scripts/_milvus_util.py —— 建库脚本共用的 Milvus 工具(ensure_indexed 抽出来防两处写歪)。

import time


def ensure_indexed(store, expected, timeout=300):
    """插入后必须 flush, 并等索引覆盖所有行, 否则检索结果不正确。
    (2026-09-10 实测)刚 insert 的行可能留在未封段 growing segment 不入索引:
      文本腿(dense+sparse 两索引)→ hybrid_search 整个漏掉这些行, 掉的是召回;
      多模态腿 → growing 命中被焊在封段结果之前不看分数, top1 分反而低于后面页。
    判据: describe_index 的 indexed_rows / pending_index_rows; 多索引逐个查,
      期望全部已覆盖 且 pending == 0。

    :param store:   带 .client / .collection_name 的 store
                    (MultimodalPageStore / MultimodalFDEStore)
    :param expected: 应被索引覆盖的行数(页数)
    :param timeout:  等待秒数; 多模态页库默认 300, FDE(秒级)传 120 即可
    :return: 覆盖完整返回 True; flush 失败 / 无索引 / 超时返回 False
    """
    client = store.client
    try:
        client.flush(collection_name=store.collection_name)
    except Exception as e:
        print(f'   [警告] flush 失败: {type(e).__name__}: {e}', flush=True)
        return False

    index_names = client.list_indexes(collection_name=store.collection_name)
    if not index_names:
        print('   [警告] 集合没有索引, 跳过覆盖率检查', flush=True)
        return False

    t = time.time()
    while time.time() - t < timeout:
        all_ok = True
        pending_report = []
        for name in index_names:
            d = client.describe_index(collection_name=store.collection_name, index_name=name)
            indexed = int(d.get('indexed_rows') or 0)
            pending = int(d.get('pending_index_rows') or 0)
            pending_report.append(f'{name}: {indexed}/{expected}(pending={pending})')
            if indexed < expected or pending != 0:
                all_ok = False
        if all_ok:
            print(f'   索引已覆盖 {expected}/{expected} 行(pending=0): '
                  f'{" | ".join(pending_report)}', flush=True)
            return True
        time.sleep(2)
    print(f'   [警告] 超时: 索引覆盖不全({"; ".join(pending_report)}), '
          f'可能有行没进索引而被漏检/乱序', flush=True)
    return False