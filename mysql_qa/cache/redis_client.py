# Redis 缓存: JSON 存取, 给 L1 答案缓存与 BM25 语料复用

import redis
import json
import sys, os

# __file__ 向上定位到项目根, 让直接运行本文件也能 import 到 base 包
current_dir = os.path.dirname(os.path.abspath(__file__))    # mysql_qa/cache
module_path = os.path.dirname(current_dir)                  # mysql_qa
project_dir = os.path.dirname(module_path)                  # 项目根
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# 别写 `from base import logger`(会被子模块导入覆写成模块对象), 写完整子模块路径:
# base/__init__.py 里绑的是日志实例, 但一旦别处执行 `from base.logger import logger`,
# base.logger 属性就被覆写成模块对象, logger.info() 会 AttributeError
from base.config import Config
from base.logger import logger


class RedisClient(object):
    """JSON 缓存存取客户端(L1 答案缓存 + BM25 语料复用)。"""

    def __init__(self):
        self.logger = logger

        try:
            conf = Config()
            # decode_responses=True 让取回的值是 str 而非 b'...', 否则和 json.loads
            # 一起用会踩"bytes 无法解析"的坑。客户端是懒连接, 首条命令才真正连。
            self.client = redis.StrictRedis(
                host=conf.REDIS_HOST,
                port=conf.REDIS_PORT,
                password=conf.REDIS_PASSWORD,
                db=conf.REDIS_DATABASE,
                decode_responses=True,
            )
            self.logger.info('redis链接成功')
        except RedisError as e:
            self.logger.error(f'redis 链接失败: {e}')
            raise

    def set_data(self, key, value, ex=None):
        """
        把可 JSON 化的对象序列化后写入。
        :param ex: 过期秒数; None=永不过期(保持老行为)。L2 回填 7 天、L3/L4 回填 1 天
                   (config.ini [cache_ttl])。ex 透传给 SET 的 EX, 由 Redis 保证过期。
        """
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False), ex=ex)
            # value 不打进日志: 答案可能很长, 全量刷入会淹掉有用信息
            self.logger.info(f'数据存储成功: {key} (ttl={ex if ex else "永久"})')
        except RedisError as e:
            self.logger.error(f'数据存储失败: {e}')
            raise

    def get_data(self, key):
        """读回 key 的 JSON 并还原成原对象; key 不存在返回 None。"""
        try:
            result = self.client.get(key)
            return json.loads(result) if result else None
        except RedisError as e:
            self.logger.error(f'获取数据失败:{key}')
            raise

    def get_answer(self, query):
        """按固定规则 answer:{query} 精确查问答缓存; 命中返回还原后的答案, 未命中返回 None。"""
        try:
            answer = self.client.get(f'answer:{query}')
            if answer:
                self.logger.info(f"从 Redis 获取答案: {query}")
                return json.loads(answer)
            return None
        except RedisError as e:
            # 缓存挂了不能让问答主流程崩 -> 当"没命中"让上层查库
            self.logger.error(f"Redis 查询失败: {e}")
            return None


if __name__ == '__main__':
    # 冒烟: python mysql_qa/cache/redis_client.py (需本地 Redis 已启动)
    redcli = RedisClient()
    print('① 连通性 ping:', redcli.client.ping())

    cache_key = 'smoke:user:1'
    redcli.set_data(cache_key, {'name': '张三', 'age': 18})
    back = redcli.get_data(cache_key)
    print('② set/get 还原:', back, '| 类型:', type(back).__name__)

    print('③ 空 key ->', redcli.get_data('smoke:not_exist'))

    redcli.set_data('answer:1+1=几', '等于 2')
    print('④ 问答缓存命中 ->', redcli.get_answer('1+1=几'))
    print('④ 问答缓存未命中 ->', redcli.get_answer('没问过的问题'))

    redcli.client.delete(cache_key, 'answer:1+1=几')
    print('⑤ 清理后 ->', redcli.get_data(cache_key))

    redcli.client.close()
