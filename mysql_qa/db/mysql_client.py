# MySQL 基础操作: 建库建表(jpkb)、灌 CSV(9755 条)、取题目清单、按题取答案

import pymysql
import pandas as pd
import sys, os

# 用 __file__ 向上跳 3 级定位项目根(db/ -> mysql_qa/ -> 项目根), 不依赖 CWD;
# 下面用包路径 base.xxx, Python 找 base 包去"base 的父目录"即项目根
current_dir = os.path.dirname(os.path.abspath(__file__))    # mysql_qa/db
module_path = os.path.dirname(current_dir)                  # mysql_qa
project_dir = os.path.dirname(module_path)                  # 项目根
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# 别写 `from base import Config, logger`: base/__init__.py 里 `from logger import logger`
# 绑的是实例, 但只要别处执行过 `from base.logger import logger`(子模块导入), Python 会把
# base.logger 属性覆写成模块对象, 于是 logger.info() 会 AttributeError。写完整子模块路径。
from base.config import Config
from base.logger import logger


class MysqlClient():
    """封装"连 MySQL / 建表 / 增删改查 / 关连接", 上层拿到实例即可干活。"""

    def __init__(self):
        self.logger = logger
        try:
            conf = Config()

            # charset 必须 utf8mb4, 不能 'utf8'(那是 utf8mb3, 每字符最多 3 字节):
            # 答案含 emoji 时写库报 (1366, "Incorrect string value...")
            self.connection = pymysql.connect(
                host=conf.MYSQL_HOST,
                port=int(conf.MYSQL_PORT),   # 容器部署 [mysql] port=13306
                user=conf.MYSQL_USER,
                password=conf.MYSQL_PASSWORD,
                db=conf.MYSQL_DATABASE,
                charset='utf8mb4',
            )

            self.cursor = self.connection.cursor()
            self.logger.info("Connected to MySQL Server")
        except pymysql.MySQLError as e:
            self.logger.error(f'mysql链接异常：{e}')
            raise

    def create_table(self):
        # 建表要点: 最后一行 CREATE 子句后不能留逗号; AUTO_INCREMENT 列必须 PRIMARY KEY;
        # 问题/答案用 MEDIUMTEXT(真实答案最长 2389 字, VARCHAR(255) 会 Data too long)。
        # category 是汽车故障类别(存 21 类故障), 目前是**死列**: 全仓只有按 question 查,
        #   没有任何按类别过滤的读端(旧 source_filter 已随 GET /api/sources 删掉)。
        #   别看到 idx_category 就反推"系统有类别过滤"。它不是旧模型的"学科类别"。
        # 改列名/加列只能 DROP 重建: CREATE TABLE IF NOT EXISTS 对已存在表是空操作,
        #   重建入口 = 本文件 __main__ 的 --force。
        create_table_sql = '''
        CREATE TABLE IF NOT EXISTS `jpkb` (
            id       BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
            category VARCHAR(64)     NOT NULL DEFAULT '',
            question MEDIUMTEXT,
            answer   MEDIUMTEXT,
            PRIMARY KEY (id),
            KEY idx_category (category)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''
        try:
            self.cursor.execute(create_table_sql)
            self.connection.commit()
            self.logger.info("表 jpkb 创建成功(若已存在则跳过)")
        except pymysql.MySQLError as e:
            self.logger.error(f'建表失败: {e}')

    def insert_data(self, csv_path):
        # 用 %s 占位符 + 参数传值而非拼 SQL, 防注入也免转义
        insert_sql = (
            "INSERT INTO `jpkb` (`category`, `question`, `answer`) "
            "VALUES (%s, %s, %s)"
        )
        try:
            data_df = pd.read_csv(csv_path)
            # 数据源表头是中文, 归一成英文列名; 换数据源时这里是第一处要核对的地方,
            # 对不上会当场 KeyError(rename 对不认识的列名不改动)
            data_df = data_df.rename(columns={
                '类别': 'category',
                '问题': 'question',
                '答案': 'answer',
            })

            for _, row in data_df.iterrows():
                self.cursor.execute(
                    insert_sql,
                    (row['category'], row['question'], row['answer']),
                )

            self.connection.commit()
            self.logger.info(f'数据插入成功: {csv_path}')
        except Exception as e:
            # 任何一步失败先回滚, 避免带着半截数据继续跑
            self.connection.rollback()
            self.logger.error(f'数据插入失败: {e}')
            raise

    def fetch_questions(self):
        try:
            self.cursor.execute('SELECT question FROM jpkb')
            results = self.cursor.fetchall()   # 元组的元组 ((q,), (q,), ...), 非 DataFrame
            self.logger.info(f'查询成功: 共 {len(results)} 条')
            return results
        except pymysql.MySQLError as e:
            self.logger.error(f'查询失败：{e}')
            return []

    def get_answer(self, question):
        """按问题原文精确查答案; 未找到/出错都返回 None, 上层统一按"没有答案"处理。"""
        try:
            # 用户/外部输入走 %s 参数, 无 SQL 注入风险
            self.cursor.execute(
                'SELECT answer FROM jpkb WHERE question=%s',
                (question,),
            )
            results = self.cursor.fetchone()   # 取第一条, 没有则 None
            return results[0] if results else None
        except pymysql.MySQLError as e:
            self.logger.error(f'查询失败: {e}')
            return None

    def close_connection(self):
        """用完要关: 只关 cursor 不算完, connection 也关, 否则占着连接池。"""
        try:
            self.cursor.close()
            self.connection.close()
            self.logger.info('MySQL 连接已关闭')
        except pymysql.MySQLError as e:
            # 关闭是收尾动作, 失败不中断上层流程
            self.logger.error(f'mysql关闭失败: {e}')


if __name__ == '__main__':
    # 演示入口: python mysql_qa/db/mysql_client.py [--force]
    #   --force = 删表重建重导 + 清 Redis 语料缓存; 默认表空才导、非空跳过
    from mysql_qa.cache.redis_client import RedisClient

    # CSV 用 module_path 锚定, 不用 CWD 相对路径(换目录启动会 FileNotFoundError, 看着像数据没生成)
    test_csv = os.path.join(module_path, 'data', '汽车售后问答.csv')
    # 默认不重建, "随手跑一次就把库清掉"不能是默认行为
    force = '--force' in sys.argv

    mysql_client = MysqlClient()
    try:
        if force:
            mysql_client.cursor.execute('DROP TABLE IF EXISTS `jpkb`')
            mysql_client.connection.commit()
            print('--force: 已删除旧表 jpkb')

        mysql_client.create_table()

        mysql_client.cursor.execute('SELECT COUNT(*) FROM `jpkb`')
        if mysql_client.cursor.fetchone()[0] == 0:
            print(f'jpkb 为空, 导入 {test_csv} ...')
            mysql_client.insert_data(test_csv)
        else:
            print('jpkb 已有数据, 跳过导入(如需清空重导: 加 --force)')

        # --force 必须一并清 Redis 语料缓存: BM25 先查 Redis 查不到才查库, 表重建后旧缓存必失效,
        #   忘清的症状是"新数据一条也检索不到, 且一句错都不报"。清: 语料 key + answer:* + qa:*。
        #   (qa:* 不匹配 qa_original_questions —— 前缀是 `qa:` 不是 `qa_`)。放在导入之后,
        #   万一导入失败至少不去动一份还能用的缓存。
        if force:
            redis_client = RedisClient()
            stale_keys = (
                ['qa_original_questions', 'qa_tokenized_questions']
                + list(redis_client.client.scan_iter('answer:*'))
                + list(redis_client.client.scan_iter('qa:*'))
            )
            # delete 返回真正删掉的条数, 打真实删除数而非尝试数
            deleted = redis_client.client.delete(*stale_keys)
            print(f'--force: 已清 Redis 旧语料/答案缓存 {deleted} 个 key'
                  f'(共尝试 {len(stale_keys)} 个)')
            redis_client.client.close()

        print('① 全部问题条数 :', len(mysql_client.fetch_questions()))
        mysql_client.cursor.execute('SELECT `question` FROM `jpkb` LIMIT 1')
        real_q = mysql_client.cursor.fetchone()[0]
        ans = mysql_client.get_answer(real_q)
        print(f'② 命中答案 : (Q={real_q[:30]}...) -> {str(ans)[:60]}...')
        print('③ 未命中   :', mysql_client.get_answer('不存在的题目'))
    finally:
        mysql_client.close_connection()
