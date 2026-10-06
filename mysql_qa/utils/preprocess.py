# 文本预处理: 小写 + jieba 分词, 供 BM25 建语料与查词共用同一词表
import jieba
# 不要写 `from base import logger`(会被子模块导入覆写成模块对象), 见 redis_client.py 说明
from base.logger import logger

def preprocess_text(data):
    logger.info('开始预处理数据')
    try:
        docs=jieba.lcut(data.lower())
        return docs
    except AttributeError as e:
        logger.error(f'数据处理失败：{data}')


if __name__ == '__main__':
    print(preprocess_text('[用户名]是最棒的'))