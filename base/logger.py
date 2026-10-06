# 日志设置: 建全局 logger(文件 + 控制台双 Handler, 统一 Formatter), 供各处 import

import logging
import os
from config import Config   # 直接运行时 Python 会把 base/ 加入 sys.path, 故顶层导入可用

# 用 __file__ 向上定位项目根, 摆脱 CWD 依赖; 日志文件名从 config.ini [logger] log_file 读
module_path = os.path.abspath(__file__)
base_dir = os.path.dirname(module_path)
current_dir = os.path.dirname(base_dir)
log_file = os.path.join(current_dir, Config().LOG_FILE)


def setup_logger(name, log_file=log_file):
    """建"文件 + 控制台"双输出 logger 并返回; name 决定日志里区分哪块业务。"""
    # 建的是目录(先 dirname), 不是文件本身 —— makedirs(log_file) 会把 app.log 建成文件夹
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 去重: 同名 logger 首次进来 handlers 才非空, 跳过已配好的; 代价是级别/路径要一次配对
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Windows 不指定 encoding 会按 GBK 写, 之后按 utf-8 读会 UnicodeDecodeError; 故显式 utf-8
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# 模块级默认 logger, 名字固定 'app'; 想按模块区分来源请各自调 setup_logger(__name__)
logger = setup_logger('app')


if __name__ == '__main__':
    logger.info('自测: 本行应同时打印到控制台并写入日志文件')
