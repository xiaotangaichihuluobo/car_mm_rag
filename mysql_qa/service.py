# L2 服务层: 包住 BM25 + 回填, 对外只暴露 lookup(query) -> QAResult。
# 命中(hit=True): 返回 jpkb 答案 + 按 TTL 回填 Redis; 未命中: 编排器丢给 L3

import os
import sys
import time

# 本文件在 mysql_qa/, __file__ 向上定位项目根
current_dir = os.path.dirname(os.path.abspath(__file__))     # mysql_qa
project_root = os.path.dirname(current_dir)                  # 项目根
for _dir in (project_root, current_dir):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from base.config import Config
from base.logger import logger
from base.qa_result import QAResult, cache_key
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MysqlClient
from mysql_qa.retrieval.bm25_search import BM25Search

conf = Config()

# 与 bm25_search.py 默认一致, 不要单独调一个
DEFAULT_THRESHOLD = 0.85


class MySQLQAService(object):
    """L2: MySQL QA 库(BM25)检索。"""

    def __init__(self, redis_client=None, mysql_client=None, bm25_search=None,
                 threshold=DEFAULT_THRESHOLD, cache_ttl=None):
        self.logger = logger
        self.threshold = threshold
        # L2 是人工标注答案, 可信度高 -> 长 TTL(默认 7 天)
        self.cache_ttl = cache_ttl if cache_ttl is not None else conf.TTL_L2_MYSQL
        # 依赖注入: 编排器已建过连接时复用, 不重复建
        self.redis_client = redis_client or RedisClient()
        self.mysql_client = mysql_client or MysqlClient()
        self.bm25_search = bm25_search or BM25Search(self.redis_client, self.mysql_client)

    def lookup(self, query):
        """
        在 MySQL QA 库里查这个问题。
        命中 source='mysql' hit=True; 未命中 hit=False 且 answer=''
        """
        t0 = time.time()
        if not query or not str(query).strip():
            return QAResult.miss('mysql', reason='empty_query')

        # search() 自带 answer:{query} 缓存与 MySQL 查询, 这里只把 (answer, need_fallback) 转成 QAResult
        try:
            answer, need_fallback = self.bm25_search.search(query, threshold=self.threshold)
        except Exception as e:
            # 检索异常不当把整个级联打挂, 记下当"未命中"继续走 L3/L4
            self.logger.error(f'L2 检索异常, 视为未命中: {e}')
            return QAResult.miss('mysql', error=str(e), latency=time.time() - t0)

        latency = time.time() - t0

        if answer and not need_fallback:
            # 这里写的 key 必须与 L1 读的 key 同一套(cache_key), 否则回填的永远读不到, L1 命中率恒 0
            self._backfill(query, answer)
            self.logger.info(f'L2 命中: 「{str(query)[:20]}」-> {str(answer)[:40]}')
            return QAResult.mysql_hit(answer, latency=latency, threshold=self.threshold)

        self.logger.info(f'L2 未命中, 交给 L3: 「{str(query)[:20]}」')
        return QAResult.miss('mysql', latency=latency, threshold=self.threshold)

    def _backfill(self, query, answer):
        """
        把 L2 答案回填到 L1 缓存。存 dict 而非裸字符串, 带上 origin 才能在命中时
        知道这条缓存是哪一层产生的, 各层命中率统计才分得清。
        """
        try:
            self.redis_client.set_data(
                cache_key(query),
                {'answer': answer, 'origin': 'mysql'},
                ex=self.cache_ttl,
            )
        except Exception as e:
            # 缓存是加速手段不是正确性依赖, 回填失败不影响本次回答
            self.logger.warning(f'L2 回填缓存失败(不影响本次回答): {e}')

    def close(self):
        """释放 MySQL 连接。Redis 连接由调用方决定是否关闭。"""
        try:
            self.mysql_client.close_connection()
        except Exception as e:
            self.logger.warning(f'关闭 MySQL 连接失败: {e}')


if __name__ == '__main__':
    # 冒烟 L2 服务层: PYTHONPATH=<项目根> python mysql_qa/service.py (需 MySQL/Redis 已启动, jpkb 有数据)
    svc = MySQLQAService()
    try:
        # 库内真题原样问 -> 应命中 L2
        svc.mysql_client.cursor.execute('SELECT question FROM jpkb LIMIT 1')
        q = svc.mysql_client.cursor.fetchone()[0].strip()
        r = svc.lookup(q)
        print(f'① 库内真题 -> hit={r.hit} source={r.source} '
              f'answer={str(r.answer)[:40]} latency={r.meta.get("latency", 0):.3f}s')
        assert r.hit and r.source == 'mysql'

        # 回填是否真写进 L1 的 key(级联能加速的前提)
        from base.qa_result import cache_key
        cached = svc.redis_client.get_data(cache_key(q))
        print(f'② L1 回填 -> {str(cached)[:60]} ...')
        assert cached and cached.get('origin') == 'mysql'

        # 回填 key 要带 TTL, 否则答案过期不更新
        ttl = svc.redis_client.client.ttl(cache_key(q))
        print(f'③ 回填 TTL -> {ttl}s (期望 >0 且 <= {svc.cache_ttl})')
        assert 0 < ttl <= svc.cache_ttl

        # 库外问题 -> 未命中且不回填(没答案可回填)
        r2 = svc.lookup('今天中午吃什么')
        print(f'④ 库外问题 -> hit={r2.hit} source={r2.source} (期望 False / mysql)')
        assert not r2.hit

        # 空问题 -> 未命中且不抛异常
        r3 = svc.lookup('   ')
        print(f'⑤ 空问题 -> hit={r3.hit} reason={r3.meta.get("reason")} (期望 False / empty_query)')
        assert not r3.hit

        svc.redis_client.client.delete(cache_key(q))
        print('⑥ 已清理自测缓存')
        print('\nL2 Service 冒烟: PASS')
    finally:
        svc.close()
        svc.redis_client.client.close()
