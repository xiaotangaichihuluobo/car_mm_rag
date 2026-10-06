# 配置文件管理: 定位 config.ini / .env, 解析各业务段配置成全局可用属性

import configparser
import os
from dotenv import load_dotenv

# 项目根 .env 的键注入 os.environ(密钥只放这里, 不进版本库)
load_dotenv()

# 基于 __file__ 向上定位项目根, 与启动目录无关
current_file_path = os.path.abspath(__file__)          # base/config.py
current_dir_path = os.path.dirname(current_file_path)  # base/
project_root_path = os.path.dirname(current_dir_path)  # 项目根
config_file_path = os.path.join(project_root_path, 'config.ini')


class Config:
    """解析 config.ini 各段键值存成 self.xxx(get(...fallback) 键缺失不抛异常, 有兜底)。"""

    def _abs(self, path):
        """把 ini 的相对路径解析成基于项目根的绝对路径; 已是绝对路径原样返回。"""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.PROJECT_ROOT, path))

    @staticmethod
    def _pdf_stem(path):
        """PDF 文件名去尾缀当默认 source_id(train_a.pdf -> 'train_a')。"""
        return os.path.splitext(os.path.basename(path))[0]

    def book_pdf(self, source_id):
        """source_id -> 该书 PDF 绝对路径; 未知书抛 KeyError 防拼错静默指向别的书。"""
        return self.MM_BOOKS[source_id]

    def book_image_dir(self, source_id):
        """source_id -> 该书页图子目录(MM_PAGE_IMAGE_DIR/{source_id}/)。"""
        return os.path.join(self.MM_PAGE_IMAGE_DIR, source_id)

    def __init__(self, config_file=config_file_path):
        # ExtendedInterpolation 支持 ${} 段间引用
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # Windows 默认按 GBK 读含中文 ini 会抛 UnicodeDecodeError, 必须显式 utf-8
        self.config.read(config_file, encoding='utf-8')

        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_PORT = self.config.get('mysql', 'port', fallback='3306')
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='123456')
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')

        # 库编号的键名是 db, 不是 database
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        self.REDIS_PORT = self.config.get('redis', 'port', fallback='6379')
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234')
        self.REDIS_DATABASE = self.config.get('redis', 'db', fallback='0')

        # 日志
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')

        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # 不复用旧 itcast/edurag_final 库, 新链路一律新建库表(家规 R2b)

        # LLM(DashScope): 只从 .env 读; config.ini 无 [llm] 段, fallback 仅兜底
        self.LLM_MODEL = os.getenv('DASHSCOPE_MODEL', 'qwen-plus')
        # 实测 qwen-vl-max / qwen-vl-plus 可用, '-latest' 后缀 id 会 403, 别用
        self.LLM_VL_MODEL = os.getenv('DASHSCOPE_VL_MODEL', 'qwen-vl-max')
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '记得改成你的密钥')
        self.DASHSCOPE_BASE_URL = os.getenv(
            'DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 多模态检索(L4): ColQwen2 页检索 + qwen-vl 生成; 相对路径统一 _abs 解析, 换目录启动才不丢文件
        self.MM_PDF_PATH = self._abs(self.config.get(
            'multimodal', 'pdf_path', fallback='rag_qa/data/car_data/train_a.pdf'))
        self.MM_PAGE_IMAGE_DIR = self._abs(self.config.get(
            'multimodal', 'page_image_dir', fallback='rag_qa/data/pages'))

        # 书目注册表([books] 每行 `source_id = pdf相对路径`)。缺段时回退单书: MM_PDF_PATH, source_id 取文件名 stem
        _books = {}
        if self.config.has_section('books'):
            for _sid, _path in self.config.items('books'):
                _path = (_path or '').strip()
                if _path:
                    _books[_sid] = self._abs(_path)
        self.MM_BOOKS = _books or {self._pdf_stem(self.MM_PDF_PATH): self.MM_PDF_PATH}
        self.MM_DEFAULT_BOOK = next(iter(self.MM_BOOKS))
        self.MM_PDF_PATH = self.MM_BOOKS[self.MM_DEFAULT_BOOK]

        self.MM_MODEL_PATH = self._abs(self.config.get(
            'multimodal', 'model_path', fallback='rag_qa/models/colqwen2_local'))
        self.MM_MODELSCOPE_ID = self.config.get(
            'multimodal', 'modelscope_id', fallback='vidore/colqwen2-v1.0-hf')
        self.MM_COLLECTION_NAME = self.config.get(
            'multimodal', 'collection_name', fallback='colqwen2_page_vectors')
        # fallback 必须与 config.ini [multimodal] 一字不差: ini 是 gitignored(含真密码),
        #   缺它的环境会**静默**落到 fallback, 指错库就不报错地检索错库。
        #   也别用 [milvus] 段的 database='itcast'(那是另一个库, 会 collection not found, R2b)。
        self.MM_MILVUS_DATABASE = self.config.get(
            'multimodal', 'database_name', fallback='car_mm')
        self.MM_LEG_TOPK = self.config.getint('multimodal', 'leg_topk', fallback=5)
        # 文字 query↔页图 MaxSim 已做查长归一(÷|Q|), 量纲 0~1; 标定(calib_mm_refuse)
        self.MM_REFUSE_SCORE = self.config.getfloat('multimodal', 'refuse_score', fallback=0.55)

        # MUVERA 两阶段图片腿(设计见 docs/superpowers/specs/2026-09-11-muvera-design.md)
        # fallback='muvera' 是默认路径; 回退只应改 ini 而非这个 fallback
        self.MM_RETRIEVAL_MODE = self.config.get(
            'multimodal', 'retrieval_mode', fallback='muvera')
        self.MM_FDE_COLLECTION_NAME = self.config.get(
            'multimodal', 'fde_collection_name', fallback='colqwen2_page_fde')
        # FDE 维度 = reps × 2^k_sim × dim_proj, 四值须与建库时完全一致(H_r/P_r 由 seed 决定;
        #   不一致=不在同一空间, 不报错但结果全错)。改了必须 --rebuild 重建 FDE 集合。
        self.MM_FDE_K_SIM = self.config.getint('multimodal', 'fde_k_sim', fallback=6)
        self.MM_FDE_DIM_PROJ = self.config.getint('multimodal', 'fde_dim_proj', fallback=16)
        self.MM_FDE_REPS = self.config.getint('multimodal', 'fde_reps', fallback=20)
        self.MM_FDE_SEED = self.config.getint('multimodal', 'fde_seed', fallback=42)
        # 阶段1 丢给阶段2 的候选页数, 同时是召回天花板: 真值页没进=救不回来
        self.MM_FDE_TOPK = self.config.getint('multimodal', 'fde_topk', fallback=50)
        # FLAT=穷举精确 MIPS, 让评估召回损失只归因 FDE 近似不掺 ANN; 语料涨了切 HNSW
        self.MM_FDE_INDEX = self.config.get('multimodal', 'fde_index', fallback='FLAT')

        # 缓存 TTL(秒): L2 人工标注可信->长; L3 直答/L4 生成->短; 拒答不回填
        self.TTL_L2_MYSQL = self.config.getint('cache_ttl', 'l2_mysql', fallback=604800)
        self.TTL_L3_DIRECT = self.config.getint('cache_ttl', 'l3_direct', fallback=86400)
        self.TTL_L4_RAG = self.config.getint('cache_ttl', 'l4_rag', fallback=86400)

        # 应用; 静态目录走 _abs 解析, 换启动目录写入点与挂载点才不分离
        self.APP_STATIC_DIR = self._abs(self.config.get('app', 'static_dir', fallback='static'))


if __name__ == '__main__':
    # 冒烟自测: python base/config.py, 从哪个目录启动都行
    def _mask(secret):
        # 密钥只留头尾, 避免明文进终端记录/日志
        return f'{secret[:4]}...{secret[-4:]}' if len(secret) > 8 else '***'

    # Config() 由 __file__ 定位 ini, 不依赖启动目录; read() 读不到是静默的, 用 sections() 确认
    cfg = Config()
    print('① 默认构造 OK, 命中的段:', cfg.config.sections())

    print(f'② [mysql]    host={cfg.MYSQL_HOST} user={cfg.MYSQL_USER} '
          f'pwd_len={len(cfg.MYSQL_PASSWORD)} db={cfg.MYSQL_DATABASE}')
    print(f'   [redis]    host={cfg.REDIS_HOST} port={cfg.REDIS_PORT} db={cfg.REDIS_DATABASE}')
    print(f'   [multimodal] mode={cfg.MM_RETRIEVAL_MODE} col={cfg.MM_FDE_COLLECTION_NAME} '
          f'k_sim={cfg.MM_FDE_K_SIM} dim_proj={cfg.MM_FDE_DIM_PROJ} reps={cfg.MM_FDE_REPS} '
          f'seed={cfg.MM_FDE_SEED} topk={cfg.MM_FDE_TOPK} index={cfg.MM_FDE_INDEX}')
    # FDE 维度必须恰为 20480; 断言把"维度对不对"和"模式名合不合法"一起卡住 ——
    # 全项目只有 FDEEncoder.dim 对应它, 算错=查询/建库不在同一空间
    _fde_dim = cfg.MM_FDE_REPS * (2 ** cfg.MM_FDE_K_SIM) * cfg.MM_FDE_DIM_PROJ
    assert _fde_dim == 20480, f'FDE 维度算出来是 {_fde_dim}, 期望 20480'
    assert cfg.MM_RETRIEVAL_MODE in ('muvera', 'maxsim'), \
        f'retrieval_mode 只能是 muvera/maxsim, 实际 {cfg.MM_RETRIEVAL_MODE!r}'
    print(f'   FDE 维度 = {_fde_dim} OK')
    print(f'   [app]      static={cfg.APP_STATIC_DIR}')

    print(f'③ [llm/env]  model={cfg.LLM_MODEL} api_key={_mask(cfg.DASHSCOPE_API_KEY)}')

    # 段/键都不存在也不抛异常, 返回默认值
    print(f'④ fallback = {cfg.config.get("no_such", "no_such", fallback="默认值")}')

    # 显式传 config_file 也能读, 应与 ① 一致
    cfg2 = Config(config_file=config_file_path)
    print(f'⑤ 显式传 config_file 构造 OK, db={cfg2.MYSQL_DATABASE}')
