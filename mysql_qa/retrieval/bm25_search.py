# L2 BM25 检索: 整库题目 softmax 打分, >=0.85 判"库里有没有这题", 命中取 jpkb 答案 + 回填缓存
from rank_bm25 import BM25Okapi
import numpy as np
import sys, os

# __file__ 向上定位到项目根, 让直接运行时也能 import 到 base / utils
current_dir = os.path.dirname(os.path.abspath(__file__))    # mysql_qa/retrieval
module_path = os.path.dirname(current_dir)                  # mysql_qa
project_dir = os.path.dirname(module_path)                  # 项目根
for _dir in (module_path, project_dir):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# 别写 `from base import logger`(会被子模块导入覆写成模块对象), 写完整子模块路径
from base.config import Config
from base.logger import logger
from utils.preprocess import preprocess_text

# 答案级缓存 TTL 口径与 service.py 一致(config.ini [cache_ttl] l2_mysql)
cfg = Config()


class BM25Search(object):
    def __init__(self, redis_client, mysql_client, cache_ttl=None):
        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        # L2 是人工标注答案, 可信度高 -> 长 TTL(默认 7 天); 千万别永不过期:
        # jpkb 改答案后缓存不变就永远返回旧的, 而日志那句"获取到正确答案"是硬编码文案, 无校验
        self.cache_ttl = cache_ttl if cache_ttl is not None else cfg.TTL_L2_MYSQL
        self.bm25 = None
        self.questions = None
        self.original_questions = None
        self._load_data()

    def _load_data(self):
        """加载(或从缓存重建)BM25 题库语料: 先查 Redis, 未命中才查库、分词后回填。"""
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"

        self.original_questions = self.redis_client.get_data(original_key)
        tokenized_questions = self.redis_client.get_data(tokenized_key)

        # 两个 key 任一为空都算未命中(首启 / 被清空 / 只写了一半)
        if not self.original_questions or not tokenized_questions:
            self.original_questions = self.mysql_client.fetch_questions()  # 元组的元组

            if not self.original_questions:
                self.logger.warning("未加载到问题")   # 空表建不了 BM25
                return

            # 与 search() 用同一 preprocess_text, 保证词表对得上
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]

            # 拍平成 list[str], 否则 search() 里 [best_idx] 取到元组而不是字符串
            self.original_questions = [q[0] for q in self.original_questions]

            self.redis_client.set_data(original_key, self.original_questions)
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info("BM25 模型初始化完成")

    def _softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        """
        对外主入口: ①BM25 相似度 -> ②答案级缓存(标准问题 key) -> ③MySQL 查答案。

        **没有**"用户原话 -> 答案"这一级缓存: 原话会被 BM25 归一成题库另一道题,
        把"那题的答案"存到"原话"key 下, 下次同句话问进来就拿到别的题的答案
        (踩过: 「1+1等于几」被归一成《实现购物车加1 减1…》, softmax 0.8917 达标,
        原话 key 永久绑定了购物车那题答案)。"同句重复问"由 L1 缓存(new_main.py)路由。

        :param query:    用户原始问题
        :param threshold: softmax 相似度阈值, 默认 0.85, 达不到视为"库里没有这题"
        :return: (answer, need_fallback); need_fallback True = 交上层兜底
        """
        if not query or not isinstance(query, str):
            self.logger.warning('无效查询')
            return None, True

        try:
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            # softmax 压到 0~1: BM25 原始分无固定范围, 不归一没法设阈值
            softmax_scores = self._softmax(scores)
            best_idx = np.argmax(softmax_scores)
            best_score = softmax_scores[best_idx]

            if best_score >= threshold:
                original_answer = self.original_questions[best_idx]

                # 日志把"归一到了哪道题"+"答案来自哪个存储"一并打出: 归一后的标准问题常与
                # 用户原话长得不一样, 不打会误以为"拿我没问过的问题答案糊弄我"
                matched = (f'BM25 归一: 「{query}」-> 「{original_answer}」 '
                           f'(softmax {best_score:.4f})')

                # 答案级缓存 key 用"标准问题": 千变万化的问法归一到同一标准题,
                # 命中率更高且 key 与值语义一致, 不串题
                redis_result = self.redis_client.get_answer(original_answer)
                if redis_result:
                    answer = redis_result
                    self.logger.info(f'{matched}, 答案来自 redis')
                else:
                    answer = self.mysql_client.get_answer(original_answer)
                    self.logger.info(f'{matched}, 答案来自 mysql')

                    if answer:
                        self.redis_client.set_data(
                            f"answer:{original_answer}", answer, ex=self.cache_ttl)
                        self.logger.info(f"搜索成功，Softmax 相似度: {best_score:.3f}")

                if answer:
                    return answer, False

            # 分数没达标 / 达标但库里没答案 -> 走兜底
            self.logger.info(f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}")
            return None, True

        except Exception as e:
            # 任一步异常都不崩进程, 记日志后走兜底
            self.logger.error(f'BM25查询失败：{query}, 异常: {e}')
            return None, True


if __name__ == '__main__':
    # 冒烟测全链路: python mysql_qa/retrieval/bm25_search.py (需 Redis/MySQL 已启动, jpkb 有数据)
    from cache.redis_client import RedisClient
    from db.mysql_client import MysqlClient

    redis_client = RedisClient()
    mysql_client = MysqlClient()
    try:
        bm25 = BM25Search(redis_client, mysql_client)
        print(f'① 冷启动完成: 共加载 {len(bm25.original_questions)} 条题目, BM25 索引已就绪')

        mysql_client.cursor.execute('SELECT question FROM jpkb LIMIT 1')
        q_std = mysql_client.cursor.fetchone()[0].strip()
        print(f'② 测试用标准题: {q_std[:50]}')
        ans, fb = bm25.search(q_std)
        print(f'   精确问 -> need_fallback={fb}, 答案前60字: {str(ans)[:60]}')

        q_para = q_std.rstrip('?？。!！ ') + '呢'
        ans, fb = bm25.search(q_para)
        print(f'③ 近似问 [{q_para[:40]}...] -> need_fallback={fb}, 答案前60字: {str(ans)[:60]}')

        ans, fb = bm25.search('今天中午吃什么')
        print(f'④ 库外问 -> need_fallback={fb}, answer={ans}   (期望 True / None)')

        ans, fb = bm25.search('')
        print(f'⑤ 空字符串 -> need_fallback={fb}, answer={ans}   (期望 True / None)')

        # 热启动: 语料缓存命中, 免重查库/重分词
        bm25_again = BM25Search(redis_client, mysql_client)
        print(f'⑥ 热启动完成: 同样加载 {len(bm25_again.original_questions)} 条题目(语料来自 Redis 缓存)')

        # 回归护栏: 「用户原话」key 不能再被写 —— 那等于把归一后那道题的答案绑到原话上
        bad = redis_client.get_answer(q_para)
        print(f'⑦ 原话「{q_para[:30]}」不应被写成答案缓存 -> {bad}   (期望 None)')

        redis_client.client.delete(
            f'answer:{q_std}', f'answer:{q_para}',
            'qa_original_questions', 'qa_tokenized_questions',
        )
        print('⑧ 已清理自测产生的 answer:* 与语料缓存 key')

    finally:
        mysql_client.close_connection()
        redis_client.client.close()
