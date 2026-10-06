# base 包 __init__: 把项目根加进 sys.path, 直接运行时才能 import 到 base

import os
import sys

# module_path = base/ 目录; project_path = 项目根(放 config.ini 的那一层)
module_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(module_path)

# 两处都插进 sys.path 前段; 先判 not in 避免重复 import 时越堆越长
if module_path not in sys.path:
    sys.path.insert(0, module_path)

if project_path not in sys.path:
    sys.path.insert(0, project_path)


# 顶层导入(靠上面的 sys.path 才能找到 base/ 里的模块)
# 坑: 若项目别处用包路径 from base.config import Config, 同一文件会被加载成两份
#   (config 与 base.config 是两个对象), isinstance / 单例判断会对不上。
from config import Config
from logger import logger
