# 主编排器 IntegratedQASystem(级联 L1->L4)所在地 + CLI 入口; app.py 只作 Web 外壳导入它

import warnings
warnings.filterwarnings("ignore")

# 导入所需的库。
from mysql_qa.db.mysql_client import MysqlClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search
from mysql_qa.service import MySQLQAService          # L2 服务层(内部包了 bm25 + 回填)

from base.config import Config
from base.logger import logger
# 统一返回协议 + 缓存 key 归一化(L1/L3/L4 都按 cache_key() 读写, 拼不到一起就永远命不中)
from base.qa_result import QAResult, cache_key
# 调用 DashScope API
from openai import OpenAI
import time
import pymysql
import uuid

confg = Config()


# 引用页在 MySQL 里存逗号分隔的「书号|页号」字符串(如 'train_a|31,train_a|39')。
#   跨书后页号不再全库唯一, 必须带书号; 候选页上限 leg_topk, VARCHAR(255) 有余量。
def _coord_str(p):
    """
    函数功能: 把各种形态的页身份归一成 canonical 字符串 'source|page'。
    认四种入参: 裸 int(31 -> 'mm_default|31') / (source,page) 元组 / [source,page] 列表 /
      已经是 'source|page' 字符串。非法值返回 None(调用方负责留痕)。
    :param p: 页身份
    :return: str 或 None
    """
    if isinstance(p, int):
        return f'{confg.MM_DEFAULT_BOOK}|{p}'
    if isinstance(p, (tuple, list)) and len(p) == 2:
        _sid, _page = p[0], p[1]
        if isinstance(_page, int):
            return f'{_sid}|{_page}'
    s = str(p).strip()
    if '|' in s:
        _sid, _p = s.split('|', 1)
        if _p.strip().isdigit():
            return f'{_sid.strip()}|{int(_p.strip())}'
    elif s.isdigit():
        return f'{confg.MM_DEFAULT_BOOK}|{int(s)}'
    return None


def _format_cited_pages(pages):
    """
    函数功能: list[页身份] -> 'train_a|31,train_a|39'。空 / None -> **None**(不是 '')。
    空值必须是 None: 落成空串后, `WHERE cited_pages IS NOT NULL` 会把"本来没页码"的行
      全算进来, 且糊掉「有页码但写失败」的信号。列是 NULLable 的, 用它。
    """
    if not pages:
        return None
    parts = [_coord_str(p) for p in pages]
    parts = [s for s in parts if s]       # 非法项丢弃走 _coord_str 的 None
    if not parts:
        return None
    return ','.join(parts)


def _parse_cited_pages(raw):
    """
    函数功能: 'train_a|31,31' -> ['train_a|31', 'mm_default|31']。None / '' -> []。
    与 _format_cited_pages 严格互逆; 老记录里的裸页号按默认书补全。
    非法项跳过而不是整串放弃(手工改过的行不该让整个历史读不出来), 但**必须留痕**。
    """
    if not raw:
        return []
    out = []
    for piece in str(raw).split(','):
        piece = piece.strip()
        if not piece:
            continue
        s = _coord_str(piece)
        if s and s not in out:
            out.append(s)
        else:
            logger.warning(f'conversations.cited_pages 有非法项, 已跳过: {piece!r} (原值 {raw!r})')
    return out


# carRAG 的核心类: 汽车售后问答系统的核心主类, 把各种工具拼起来干活。
class IntegratedQASystem:
    def __init__(self):
        self.logger = logger
        self.config = confg
        self.mysql_client = MysqlClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        try:
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY, base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            raise

        # L2 服务层: 复用上面建好的 Redis/MySQL 连接, 不重复建(系统是全局单例, 只建一次)
        self.mysql_service = MySQLQAService(
            redis_client=self.redis_client,
            mysql_client=self.mysql_client,
            bm25_search=self.bm25_search,
        )

        # L3/L4 组件一律**延迟创建**(None 占位, 首次用到才建): L3 BERT 占显存;
        #   L4 连 Milvus + 首次检索载入 ColQwen2(约 11s)。启动即建让 app.py 起服务卡十几秒,
        #   且 ColQwen2 和 BERT 抢那 4GB 显存。文本 RAG 链路(RAGSystem.__init__ 会加载 BERT
        #   到显存)不再接入级联, 级联 L4 走多模态链路(README 定稿)。
        self._classifier = None
        self._multimodal = None
        self._llm = None
        # 多模态初始化失败的"熔断标记": Milvus 挂了时不该每个请求重试一遍并打一堆栈
        self._multimodal_error = None

        self.init_conversation_table()

    # 延迟初始化: L3 意图识别分类器
    @property
    def classifier(self):
        """BERT 查询分类器(通用知识 / 专业咨询)。首次访问时才加载权重。"""
        if self._classifier is None:
            from rag_qa.core.query_classifier import QueryClassifier
            self._classifier = QueryClassifier()
        return self._classifier

    # 延迟初始化: L4 多模态问答服务
    @property
    def multimodal(self):
        """
        多模态问答服务。首次访问时才连 Milvus 并准备本地 ColQwen2。
        初始化失败时返回 None(并记下原因), 由调用方降级成拒答 ——
        手册库没建好不该让整个问答服务起不来。
        """
        if self._multimodal is None and self._multimodal_error is None:
            try:
                from rag_qa.core.multimodal_qa import MultimodalQAService
                self._multimodal = MultimodalQAService(redis_client=self.redis_client)
                self.logger.info('L4 多模态服务就绪')
            except Exception as e:
                self._multimodal_error = str(e)
                self.logger.error(f'L4 多模态服务初始化失败, 后续请求直接走拒答: {e}')
        return self._multimodal

    # 延迟初始化: LLM 客户端(L3 直答用 qwen-plus, L4 生成用 qwen-vl-max)
    @property
    def llm_client(self):
        if self._llm is None:
            from rag_qa.core.llm_client import get_llm_client
            self._llm = get_llm_client()
        return self._llm

    def init_conversation_table(self):
        """
        函数功能: 初始化 MySQL 的 conversations 表, 存储对话历史。
            CREATE 语句带齐所有列 —— 全新环境的路径; 既有库靠 _ensure_conversation_columns() 补列。
        """
        try:
            self.mysql_client.cursor.execute("""
                create table if not exists conversations(
                    id INT AUTO_INCREMENT PRIMARY KEY,          # 主键id
                    session_id VARCHAR(36) NOT NULL,            # 会话id
                    question TEXT NOT NULL,                     # 问题
                    answer TEXT NOT NULL,                       # 答案
                    timestamp DATETIME NOT NULL,                # 时间戳
                    cited_pages VARCHAR(255) NULL,              # 本次回答引用的手册页号(0 基, 逗号分隔)
                    INDEX idx_session_id (session_id)           # 创建索引列, 目的: 提高查询的效率.
                )
            """)
            self.mysql_client.connection.commit()
            self.logger.info("对话历史表初始化成功")
            self._ensure_conversation_columns()
        except pymysql.MySQLError as e:
            self.logger.error(f"初始化对话历史表失败: {e}")
            raise

    # 给**已存在**的 conversations 表补新列(幂等)。
    def _ensure_conversation_columns(self):
        """
        函数功能: 缺哪列补哪列。
        表已存在时 create table 是**整条空操作**, 新列不会出现(读端 select 报 Unknown column)。
        不能 `ADD COLUMN IF NOT EXISTS`(那是 MariaDB 语法, MySQL 8 报语法错)。
        ALTER 拿排他元数据锁, @@lock_wait_timeout 默认一年; 会话级压到 30s, 超时**原样抛出**
        拒绝启动 —— 缺列是读写的先决条件, 带坏 schema 启动比拒绝启动更糟。
        """
        self.mysql_client.cursor.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = 'conversations'
        """, (self.config.MYSQL_DATABASE,))
        existing = {row[0] for row in self.mysql_client.cursor.fetchall()}

        added = []
        for name, ddl in (('cited_pages', 'VARCHAR(255) NULL'),):
            if name not in existing:
                # 会话级生效, 只在真要 ALTER 时才发 —— 列都齐时不白白多一次往返
                self.mysql_client.cursor.execute('SET SESSION lock_wait_timeout = 30')
                # 列名与类型都是本文件写死的字面量, 不来自外部输入 -> 无注入面
                try:
                    self.mysql_client.cursor.execute(
                        f'ALTER TABLE conversations ADD COLUMN {name} {ddl}')
                except pymysql.MySQLError as e:
                    # 典型就是 1205 Lock wait timeout exceeded; raise 拒绝启动, 不跳过
                    self.logger.error(
                        f'conversations 补列 {name} 失败(可能有别的连接持锁), 终止启动: {e}')
                    self.mysql_client.connection.rollback()
                    raise
                added.append(name)
        self.mysql_client.connection.commit()
        if added:
            self.logger.info(f'conversations 补列: {added}')


    # 获取最近对话历史: 从 MySQL 查询指定会话的最近若干轮。
    def _fetch_recent_history(self, session_id, limit=5, before_id=None):
        """
        函数功能: 取最近 limit 轮对话(时间正序)。
        :param session_id: 会话唯一标识
        :param limit: 取多少轮。**喂 prompt 的调用点固定传 5**(见 answer())
        :param before_id: 只取 id 小于它的行(向上翻页游标)。None = 从最新开始
        :return: list[dict], 键是 id / question / answer / timestamp / cited_pages
                 (cited_pages 是 list[str], 形如 'source|page'; 老记录为 NULL -> [])
        """
        try:
            # ORDER BY id 而不是 timestamp: 同秒并列时按时间排序不稳定, 翻页会漏/重复(id 单调唯一)
            sql = """
                 SELECT id, question, answer, timestamp, cited_pages
                 FROM conversations
                 WHERE session_id = %s
                 """
            params = [session_id]
            if before_id is not None:
                sql += ' AND id < %s'
                params.append(before_id)
            sql += ' ORDER BY id DESC LIMIT %s'
            params.append(limit)

            self.mysql_client.cursor.execute(sql, tuple(params))
            rows = self.mysql_client.cursor.fetchall()
            history = [{'id': r[0], 'question': r[1], 'answer': r[2],
                        'timestamp': r[3],
                        'cited_pages': _parse_cited_pages(r[4])} for r in rows]
            return history[::-1]
        except pymysql.MySQLError as e:
            self.logger.error(f"获取对话历史失败: {e}")
            return []
        finally:
            # 只读也要收尾: MysqlClient 建连时**没传 autocommit**(实测 0), 否则 SELECT 一直留事务,
            #   metadata_locks 常驻 SHARED_READ 锁, 会把 ALTER 静默挂起一年。**别改 MysqlClient
            #   autocommit**(影响 mysql_qa/** 全部调用方); 包 try 让收尾失败不盖掉上面真正的异常。
            try:
                self.mysql_client.connection.commit()
            except pymysql.MySQLError as e:
                self.logger.warning(f"释放历史查询事务失败(不影响本次结果): {e}")


    # 获取会话历史: 给接口层翻页用。
    def get_session_history(self, session_id, limit=20, before_id=None):
        """
        函数功能: 取一页会话历史(时间正序), 给前端向上翻页用。
        :param session_id: 会话唯一标识
        :param limit: 本页最多多少轮
        :param before_id: 游标 —— 只取 id 小于它的行; None = 从最新开始
        :return: {'history': [...], 'has_more': bool}(has_more = 还有更旧的轮次)
        返回形态是 dict, 唯一生产调用点(app.py 的 /api/history)与 main() 演示都已同步取法。
        """
        # 多取一条用来判 has_more —— 比额外一条 COUNT(*) 便宜, 不与翻页游标错位
        rows = self._fetch_recent_history(session_id, limit=limit + 1, before_id=before_id)
        has_more = len(rows) > limit
        return {'history': rows[-limit:] if has_more else rows, 'has_more': has_more}

    # 会话列表: 给前端左侧"历史会话"面板用。
    def list_sessions(self, limit=50):
        """
        函数功能: 列出会话, 按最后活动时间倒序。
        :param limit: 最多返回多少个会话
        :return: list[dict], 键是 session_id / turns / last_at / first_question
        首句用相关子查询而非 GROUP_CONCAT: 后者默认 1024 字节静默截断, 截断后"首句"半句且不报错。
        """
        try:
            self.mysql_client.cursor.execute("""
                 SELECT c.session_id,
                        COUNT(*)         AS turns,
                        MAX(c.timestamp) AS last_at,
                        (SELECT c2.question FROM conversations c2
                          WHERE c2.session_id = c.session_id
                          ORDER BY c2.id ASC LIMIT 1) AS first_question
                 FROM conversations c
                 GROUP BY c.session_id
                 ORDER BY last_at DESC
                 LIMIT %s
                 """, (limit,))
            return [{'session_id': r[0], 'turns': r[1],
                     'last_at': r[2], 'first_question': r[3]}
                    for r in self.mysql_client.cursor.fetchall()]
        except pymysql.MySQLError as e:
            self.logger.error(f"获取会话列表失败: {e}")
            return []
        finally:
            # 同上: 只读也要收尾, 否则留下 SHARED_READ 元数据锁把 ALTER 静默挡一年
            try:
                self.mysql_client.connection.commit()
            except pymysql.MySQLError as e:
                self.logger.warning(f"释放会话列表查询事务失败(不影响本次结果): {e}")


    # 更新会话历史: 把新对话记录存入 MySQL。
    def update_session_history(self, session_id: str, question: str, answer: str,
                               cited_pages=None):
        """
        函数功能: 更新会话历史到 MySQL。
        :param session_id: 会话唯一标识
        :param question: 用户的问题
        :param answer: 系统生成的答案
        :param cited_pages: 本次回答引用的手册页号 list[int](0 基; 没有传 None/[])
        :return: 最近 5 轮历史(list, 时间正序)
        不在写入端裁剪旧记录(早期的 DELETE LIMIT 已去掉): 历史面板要向上翻看更旧轮次;
          喂 L3/L4 prompt 的仍是最近 5 轮 —— 那个限制在 answer() 读取端。
        """
        try:
            self.mysql_client.cursor.execute("""
                 INSERT INTO conversations
                     (session_id, question, answer, cited_pages, timestamp)
                 VALUES (%s, %s, %s, %s, NOW())
                 """, (session_id, question, answer, _format_cited_pages(cited_pages)))
            self.mysql_client.connection.commit()
            # 必须在 commit() **之后**取历史: _fetch_recent_history 的 finally 自带一次 commit,
            #   若跑在 commit 前会**顺手提交掉刚 INSERT 的行**, 下面 rollback 就撤不回它。
            history = self._fetch_recent_history(session_id)
            self.logger.info(f"会话 {session_id} 历史更新成功")
            return history
        except pymysql.MySQLError as e:
            self.logger.error(f"更新会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            raise
        except Exception as e:
            self.logger.error(f"更新会话历史意外错误: {e}")
            self.mysql_client.connection.rollback()
            raise


    # 清除会话历史: 删除指定会话的所有记录。
    def clear_session_history(self, session_id: str) -> bool:
        """
        函数功能: 清除指定会话历史。
        :param session_id: 会话唯一标识
        :return: True -> 清除成功, False -> 清除失败
        """
        try:
            self.mysql_client.cursor.execute("""
                 DELETE
                 FROM conversations
                 WHERE session_id = %s
                 """, (session_id,)
            )
            self.mysql_client.connection.commit()
            self.logger.info(f"会话 {session_id} 历史已清除")
            return True
        except pymysql.MySQLError as e:
            self.logger.error(f"清除会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            return False


    # 引用页码归一化: 两个键名 -> 一个 canonical 键。
    @staticmethod
    def _pick_cited_pages(result):
        """
        函数功能: 把 QAResult 里的引用页归一成 list[str]('source|page'), **拒答一律返回 []**。
        :return: list[str] —— 恒为列表, 不是 None
        两个键名都得认, 它们同源但不同时出现:
          'cited'       —— L4 当场生成路径(multimodal_qa.py)
          'cited_pages' —— L1 缓存命中路径(_l1_lookup 条件键); multimodal_qa._backfill 回填
                             Redis 也写 {'cited_pages': [...]}。只认一个键的现象:
                           「第一次有页码, 再问一次页码消失」。
        拒答清空: multimodal_qa 拒答分支带着 candidates+cited 一起返回, 正文已写"没找到",
          再挂"依据第X页"自相矛盾。**判据用 source 不是 hit**: refuse() 返回 hit=True
          (拒答是 L4 正常产出), 真实日志 26 次 refuse 全部 hit=True, 拿 hit 判 = 拒答还挂引用页。
        返回 [] 而非 None: 让「拒答」「老数据 NULL」「L2/L3 无页码」在接口层长得一模一样。
        """
        if result is None or getattr(result, 'source', None) == 'refuse':
            return []
        meta = getattr(result, 'meta', None) or {}
        raw = None
        for key in ('cited', 'cited_pages'):     # 顺序固定: 先认生成路径, 再认缓存路径
            if meta.get(key):
                raw = meta[key]
                break
        if not isinstance(raw, (list, tuple)):
            return []
        out = []
        for p in raw:
            s = _coord_str(p)
            if s and s not in out:
                out.append(s)
        if len(out) != len(raw):
            # 丢过东西就必须留痕 —— 页码条少一格而无人知道, 是本项目最贵的那类 bug
            logger.warning(f'引用页里有非法值, 已丢弃: 原 {raw!r} -> 留 {out!r}')
        return out

    # 对话历史归一化 -> 供 L3/L4 的 prompt 使用。
    @staticmethod
    def _history_to_text(history):
        """
        函数功能: 把 _fetch_recent_history() 的 [{'question':..,'answer':..}] 拼成一段文本。
        :param history: list[dict] 或 None
        :return: str, 无历史返回空串(单轮时 prompt 的 {history} 就是个空位)
        """
        if not history:
            return ''
        lines = []
        for turn in history:
            if isinstance(turn, dict):
                lines.append(f"用户: {turn.get('question', '')}\n助手: {turn.get('answer', '')}")
            else:
                lines.append(f"用户: {turn[0]}\n助手: {turn[1] if len(turn) > 1 else ''}")
        return '\n'.join(lines)

    # 四级级联主入口: 一次问答走完 L1 -> L2 -> L3 -> L4 -> 拒答.
    #
    #   层级与命中标记:
    #     L1 Redis 精确缓存  -> source='redis'
    #     L2 MySQL 题库     -> source='mysql'
    #     L3 BERT 意图识别  -> 通用知识: source='direct' 走 qwen-plus 直答; 专业咨询: 下沉 L4
    #     L4 多模态 RAG     -> source='rag' 或 'refuse'
    #     兜底              -> source='refuse'
    #
    # 查询只有文字(单通道多模态检索), 无图片维。(原"带图提问路径"已随方向修正撤除)
    def answer(self, query, session_id=None):
        """
        函数功能: 四级级联问答(非流式), 返回统一协议 QAResult。
        :param query: 用户问题(纯文本)
        :param session_id: 会话 ID, 有则读写对话历史
        :return: QAResult
        """
        start_time = time.time()
        query = (query or '').strip()
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")

        if not query:
            return QAResult.refuse('请输入您的问题。', stage='input', reason='empty_query')

        # 对话历史(有 session_id 才读)。喂 prompt 固定 5 轮, 走私有方法直连, 不经过给前端
        #   翻页的 get_session_history() —— 否则前端面板翻得越宽, L3/L4 输入就越长(静默行为变更)
        history = self._fetch_recent_history(session_id, limit=5) if session_id else []
        history_text = self._history_to_text(history)

        result = None

        # ---- L1: Redis 精确缓存 ----
        result = self._l1_lookup(query)

        # ---- L2: MySQL 题库(BM25) ----
        if result is None:
            l2 = self.mysql_service.lookup(query)
            if l2.hit:
                result = l2

        # ---- L3: BERT 意图识别 ----
        if result is None:
            result = self._l3_answer(query, history_text=history_text)

        # ---- 专业咨询没在 L3 出答案 -> 下沉 L4 ----
        if result is None:
            result = self._l4_answer(query, history_text=history_text)

        # 兜底: 任何一层都没给可用结果时统一拒答(不能返回 None 让上层崩)
        if result is None:
            result = QAResult.refuse(
                '抱歉，暂时无法回答这个问题，请联系人工客服确认。',
                stage='cascade', reason='no_layer_hit')

        # 记元信息 + 落历史。引用页码在这里归一**一次**, 之后落库与 WS 收尾帧都读
        #   result.meta['cited_pages'] 这一个值, 避免"当次有页码、刷新后没有"的不一致。
        result.meta['cited_pages'] = self._pick_cited_pages(result)
        result.meta.setdefault('elapsed', round(time.time() - start_time, 3))
        if session_id and result.hit and result.answer:
            try:
                self.update_session_history(session_id, query, result.answer,
                                            cited_pages=result.meta['cited_pages'])
            except Exception as e:
                # 历史写失败不能让这次已经算好的答案丢掉
                self.logger.error(f'写入会话历史失败(不影响本次回答): {e}')

        self.logger.info(f"级联完成: source={result.source} hit={result.hit} "
                         f"耗时={result.meta.get('elapsed')}s")
        return result

    # 只跑 L1 + L2 的短路查询: 给非流式接口用, 不是完整级联。
    def lookup_text_cached(self, query):
        """
        函数功能: 只走 L1(Redis 缓存) + L2(MySQL 题库), 命中返回 QAResult, 都未命中返回 None。
        单独开一个方法而非让接口层去摸 _l1_lookup / mysql_service: 非流式接口承载不了 L3/L4
          (L3 出网、L4 23~45s), 那两层归 WebSocket 路径; 但绕 L3/L4 ≠ 绕 L1 —— L1 毫秒级,
          跳过它这条接口的缓存永远建不起来。L2 内部已按 7 天 TTL 回填 L1, 接口层不用管缓存。
        :param query: 用户问题(纯文本)
        :return: QAResult(命中) 或 None(调用方决定降级话术)
        """
        result = self._l1_lookup(query)
        if result is None:
            l2 = self.mysql_service.lookup(query)
            if l2.hit:
                result = l2
        return result

    # L1: Redis 精确缓存查询。
    def _l1_lookup(self, query):
        """
        函数功能: L1 查缓存。
        :return: QAResult(命中) 或 None(未命中/缓存不可用 -> 交给调用方)
        """
        try:
            cached = self.redis_client.get_data(cache_key(query))
        except Exception as e:
            # 缓存挂了不是正确性问题, 当作没命中继续往下走
            self.logger.warning(f'L1 缓存读取失败, 视为未命中: {e}')
            return None

        if not cached:
            return None
        # 回填统一存 dict(带 origin), 但历史上可能存过裸字符串, 两种都要认
        if isinstance(cached, dict):
            answer = cached.get('answer')
            origin = cached.get('origin', 'unknown')
            cited = cached.get('cited_pages')
        else:
            answer, origin, cited = cached, 'unknown', None

        if not answer:
            return None
        # L4 回填时还写了引用页号 -> 透进 meta, 否则"命中缓存"与"刚算出来"前端看起来完全一样
        extra = {'cited_pages': cited} if cited else {}
        self.logger.info(f'L1 命中(原始来源 {origin})')
        return QAResult.redis_hit(answer, origin=origin, **extra)

    # L3: BERT 意图识别 + 通用知识直答。
    def _l3_answer(self, query, history_text=''):
        """
        函数功能: L3 意图识别。
            「通用知识」-> 直接问 qwen-plus, 返回 source='direct' 的 QAResult;
            「专业咨询」-> 返回 None, 由调用方下沉 L4。
            分类器不可用时按「专业咨询」处理(宁可走检索, 不要拿通用知识直答糊手册问题)。
        :return: QAResult 或 None
        """
        try:
            label, confidence, _ = self.classifier.predict_proba(query)
        except Exception as e:
            self.logger.error(f'L3 意图识别失败, 按专业咨询处理: {e}')
            return None

        self.logger.info(f'L3 意图识别: {label} (置信度 {confidence:.3f})')

        if label != '通用知识':
            return None

        from rag_qa.core.llm_client import direct_prompt
        prompt = direct_prompt().format(history=history_text, question=query)
        raw = self.llm_client.chat_text(prompt)

        # llm_client 失败时返回 "错误: xxx" 哨兵 -> 不当答案用, 下沉 L4 再试
        from rag_qa.core.llm_client import ERROR_PREFIX
        if not raw or raw.startswith(ERROR_PREFIX):
            self.logger.warning(f'L3 直答失败({str(raw)[:80]}), 下沉 L4')
            return None

        result = QAResult.direct_answer(raw, intent=label, confidence=round(confidence, 4))
        # 回填 L1: 多轮场景跳过 —— 缓存 key 不含历史, 同一句话在不同历史下答法可能不同
        if not history_text.strip():
            self._backfill_l1(query, raw, 'l3_direct', ttl=self.config.TTL_L3_DIRECT)
        return result

    # L4: 多模态检索问答。
    def _l4_answer(self, query, history_text=''):
        """
        函数功能: L4 多模态问答(内部含三道拒答闸门, 见 multimodal_qa.py)。
        查询纯文字; VL 读检索命中页的页图作答。
        多模态服务不可用(Milvus 没起 / 手册库没建 / 显存不足)时返回拒答, 不抛异常 ——
        这类环境问题在开发机很常见, 不该让整个 Web 服务 500。
        :return: QAResult
        """
        svc = self.multimodal
        if svc is None:
            self.logger.error(f'L4 不可用, 直接拒答: {self._multimodal_error}')
            return QAResult.refuse(
                '抱歉，图文问答服务当前不可用，请联系人工客服。',
                stage='l4', reason='service_unavailable', error=self._multimodal_error)
        return svc.answer(query, history=history_text)

    # L1 回填(供 L3 使用; L2/L4 各自在自己模块里回填)。
    def _backfill_l1(self, query, answer, origin, ttl=None):
        """
        函数功能: 把某层的答案写回 L1 缓存。
        :param origin: 标记这条缓存是哪层产生的('l3_direct' / 'l4_rag')
        :param ttl: 过期秒数(None = 永不过期)。调用方务必传, 否则缓存永不更新
        """
        try:
            self.redis_client.set_data(
                cache_key(query), {'answer': answer, 'origin': origin}, ex=ttl)
        except Exception as e:
            self.logger.warning(f'L1 回填失败(不影响本次回答): {e}')

    # 流式接口三元组版(比 query() 多带一个收尾附件) -> 给 WebSocket 用。
    def query_with_meta(self, query, session_id=None):
        """
        函数功能: query() 的三元组版本 —— 逐段吐出答案, **最后一块多带一个附件**。
        :return: 生成器, yield (文本块, 是否结束, extra)
                 extra 在**非收尾**每次 yield 都是 None; 只在最后一次是:
                     {'cited_pages': [int, ...]}   # 本次回答引用的页(0 基), 没有就是 []
        cited_pages 是 answer() 归一好的那一个值(result.meta['cited_pages']), 不是在这里现取的。
        """
        result = self.answer(query, session_id=session_id)
        extra = {'cited_pages': result.meta.get('cited_pages') or []}
        text = result.answer or ''

        # 按约 24 字一块切; 太碎让前端 setState 暴增, 太整没有打字机效果
        CHUNK = 24
        for i in range(0, len(text), CHUNK):
            yield text[i:i + CHUNK], False, None
        yield '', True, extra

    # 流式接口(保持原有 (token, is_complete) 协议) —— query_with_meta 的二层外壳。
    def query(self, query, session_id=None):
        """
        函数功能: 兼容 app.py WebSocket 的流式入口 —— 逐段吐出, 最后 yield ("", True) 收尾。
        级联各层(尤其 L4 证据自评)必须先拿到**完整**答案才决定给不给, 所以这里不是真流式,
          而是把已算好的整段答案切块吐出去。需要引用页码的调用方用 query_with_meta()。
        :return: 生成器, yield (文本块, 是否结束)
        """
        for token, is_complete, _extra in self.query_with_meta(
                query, session_id=session_id):
            yield token, is_complete

    # 取上一次问答的完整结果(含 source/meta), 供接口层回给前端。
    def answer_with_meta(self, query, session_id=None):
        """函数功能: answer() 的语义化别名, 强调返回值带 _meta(接口层把它透给前端)。"""
        return self.answer(query, session_id=session_id)


# main: 命令行交互界面, 测试 carRAG 问答系统。
def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("\n欢迎使用 carRAG 智能问答系统！")
    print(f"会话ID: {session_id}")
    print("输入查询进行问答，输入 'exit' 退出。")

    try:
        while True:
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见, 感谢您的使用, 期待下次再会！")
                break
            print("\n答案: ", end="", flush=True)
            answer = ""
            for token, is_complete in qa_system.query(query, session_id=session_id):
                if token:
                    print(token, end="", flush=True)
                    answer += token
                if is_complete:
                    print()
                    break
            history = qa_system.get_session_history(session_id)['history']
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        # 注意: MysqlClient 的方法名是 close_connection(), 没有 close() -> 写错会 AttributeError
        qa_system.mysql_client.close_connection()


if __name__ == '__main__':
    main()