# L4 多模态问答编排: 候选页 -> 拒答闸门 -> chat_vl 生成; 命中回填 Redis, 拒答不回填。
# 位置: 级联最内层(CLAUDE.md A2); 查询纯文字, VL 读命中页页图作答。
#
# 【拒答设计: 两道闸门(2026-09-29 阈值闸门已去掉)】
#   闸门1 无候选      —— 检索没召回任何页(没东西没法答)
#   闸门2 模型自评不足 —— qwen-vl 读完页图后说"证据不足"(见 llm_client.vl_prompt)
#   曾经的闸门2(原始 MaxSim 分 < MM_REFUSE_SCORE)已按用户裁决删除: 有候选即送生成,
#   答不答交给模型自评兜底。宁可让模型判断, 不用我们拍死的分数卡。
# ============================================================

import os
import sys

# ---- 路径引导: 把项目根放进 sys.path, 让本文件既能被 import, 也能直接 python 运行 ----
# 本文件在 rag_qa/core/, 往上退 2 层到项目根。
#
# 【为什么不把 rag_qa/ 自己也插进 sys.path】
#   把 rag_qa/ 放进 sys.path 意味着它下面的模块可以被当顶层模块导入 ——
#   这正是 _archive/README.md 记录的「rag_qa/config.py 遮蔽 base/config.py」那类
#   顺序依赖炸弹的温床。只保留 project_root。(家规 A8)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time

from base.config import Config
from base.logger import logger
from base.qa_result import QAResult, cache_key
from rag_qa.core.image_leg import PageRetriever, page_image_path
from rag_qa.core.llm_client import ERROR_PREFIX, get_llm_client, vl_prompt

conf = Config()

# 证据自评不足时, 统一话术(与 llm_client.vl_prompt 里要求模型输出的那句保持一致)
REFUSE_TEXT = '抱歉，用户手册中没有找到相关信息，建议联系客服确认。'


def parse_vl_output(text):
    """
    解析 qwen-vl 按「证据/页码/回答」格式产出的文本。

    页码值现在带书号(`train_a|31` 或裸 `31`)—— 跨书后引用必须分得清是哪本书的页。
    :param text: 模型原始输出
    :return: (evidence_sufficient: bool, pages: list[(source_id, 页码)], answer: str)
    """
    if not text or text.startswith(ERROR_PREFIX):
        return False, [], text or ''

    sufficient = None
    pages = []
    answer_lines = []
    in_answer = False

    for raw_line in str(text).splitlines():
        line = raw_line.strip()

        # 拆开 '标签: 值'; 中文/英文冒号都认, 无冒号时 label 为空
        if '：' in line:
            label, value = line.split('：', 1)
        elif ':' in line:
            label, value = line.split(':', 1)
        else:
            label, value = '', line
        label = label.strip()
        value = value.strip()

        if label.startswith('证据'):
            sufficient = '不足' not in value
            # 模型偶尔漏掉"回答:"标签直接写正文, 这里放开展收正文, 免得正文整段被丢
            in_answer = True
        elif label.startswith('页码'):
            pages = []
            for tok in value.replace('，', ',').split(','):
                tok = tok.strip()
                if not tok or tok in ('无', '无。', 'none', 'None'):
                    continue
                if '|' in tok:
                    _sid, _p = tok.split('|', 1)
                    if _p.strip().isdigit():
                        pages.append((_sid.strip(), int(_p.strip())))
                elif tok.isdigit():
                    # 裸页号: 模型没跟上新协议时的兜底, 落到默认书
                    pages.append((conf.MM_DEFAULT_BOOK, int(tok)))
            in_answer = False
        elif label.startswith('回答'):
            if value:
                answer_lines.append(value)
            in_answer = True
        elif in_answer and line:
            answer_lines.append(line)

    answer = '\n'.join(answer_lines).strip()

    # 模型没给"证据"行时的兜底: 整段当答案, 只记告警; 不因格式问题就拒答。
    #   先判 answer 为空再整体回退 —— 模型完全不写标签时循环收不进来, 只能整段取回。
    if sufficient is None:
        logger.warning(f'qwen-vl 输出未含"证据"行, 按格式外处理: {str(text)[:120]}')
        answer = answer or str(text).strip()
        return bool(answer), pages, answer

    return sufficient, pages, answer


class MultimodalQAService(object):
    """
    职责: L4 多模态问答的对外入口。输入问题(和可选的用户实拍图), 输出 QAResult。

    【本类没有 close()】生产路径(new_main.py)从不 close, 加它只会变成又一个无人调用的门。
      真要加回优雅退出, 请**新开一个对外的门**, 不要再从服务里伸手掏 store 的私有属性(家规 A9)。
    """

    def __init__(self, retriever=None, llm=None, redis_client=None, load_on_init=True):
        """
        :param retriever: PageRetriever; 不传则自建(会连 Milvus + 首次检索时载入 ColQwen2)
        :param llm: LLMClient; 不传则用进程内单例
        :param redis_client: RedisClient; 传了才做回填(测试可传 None 跳过缓存)
        :param load_on_init: 是否在构造时把 collection 载入内存
        """
        self.logger = logger
        self.retriever = retriever or PageRetriever(load_on_init=load_on_init)

        # LLM: 允许注入(测试时传假的, 不出网); 不传就用进程内单例
        self.llm = llm or get_llm_client()

        self.redis_client = redis_client
        self.prompt = vl_prompt()
        # 阈值拒答已去掉(2026-09-29 用户裁决: 阈值不限)。有候选即送生成, 答不答由闸门3
        #   模型自评兜底 —— 不再用 MM_REFUSE_SCORE 按 MaxSim 分卡。config 里的键留着备用。
        self.cache_ttl = conf.TTL_L4_RAG

    # ------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------
    def answer(self, query, history=None):
        """
        函数功能: L4 多模态问答。查询纯文字, VL 读检索命中页的页图作答。
        :param query: 用户问题(文本)
        :param history: 归一化后的历史文本(str); 仅用于填 prompt, 不参与检索
        :return: QAResult, source 为 'rag' 或 'refuse'
        """
        start = time.time()
        self.logger.info(f'[L4] 开始多模态问答: {query!r}')

        # ---- ① 检索(单通道: 文字 -> muvera+maxsim -> 页) ----
        try:
            candidates = self.retriever.retrieve(query=query)
        except Exception as e:
            # 检索挂了不该让用户看到栈: 走拒答, 让上层继续兜底
            self.logger.error(f'[L4] 多模态检索失败: {e}')
            return QAResult.refuse(REFUSE_TEXT, stage='l4', reason='retrieve_error', error=str(e))

        # ---- 闸门1: 无候选 ----
        if not candidates:
            self.logger.info('[L4] 闸门1 触发: 单通道无召回 -> 拒答')
            return QAResult.refuse(REFUSE_TEXT, stage='l4', reason='no_candidate')

        # 无分数阈值闸门(2026-09-29 用户裁决: 阈值拒答去掉)。有候选即送生成;
        #   答不答由闸门3 模型自评兜底。best_score 仍记进 meta 供评估/审计。
        best_score = max((float(c.score) for c in candidates), default=0.0)
        self.logger.info(f'[L4] 召回 {len(candidates)} 页, '
                         f'top1=p{candidates[0].page}, 最高原始分={best_score:.2f}')

        # ---- ② 生成 ----
        # 一候选一页图当证据。单通道检索每候选恒为一页(cand.page), 无跨页 span。
        images = []

        # 只送真的存在的页图; 记下实际送出去的身份(书+页), 供闸门3 的引用页校验。
        #   引用要带书号, 所以身份必须是 (source_id, page), 不能只要页号 ——
        #   跨书后"p5"分不清是哪本书的。
        pages_sent = []
        for cand in candidates:
            path = page_image_path(cand.source_id, cand.page)
            if os.path.exists(path):
                images.append(path)
                pages_sent.append((cand.source_id, cand.page))
            else:
                self.logger.warning(f'[L4] 页图缺失, 跳过: {path}')

        if not pages_sent:
            self.logger.error('[L4] 候选页图全部缺失 -> 拒答')
            return QAResult.refuse(REFUSE_TEXT, stage='l4', reason='page_image_missing',
                                   candidates=[cand.key for cand in candidates])

        # 查询纯文字, 恒有值; 空 query 早在闸门1(无召回)就被拒, 到不了这里。
        question_text = str(query or '').strip()

        # 候选页身份清单送进 prompt —— 模型要引用"书+页", 就必须先知道每张页图属于哪本书
        pages_desc = ', '.join(f'{sid}|{p}' for sid, p in pages_sent)
        prompt_input = self.prompt.format(history=history or '', question=question_text,
                                          pages_desc=pages_desc)
        raw = self.llm.chat_vl(prompt_input, images=images)

        # ---- 闸门3: 模型自评证据不足 ----
        sufficient, cited_pages, answer_text = parse_vl_output(raw)
        if not sufficient or not answer_text:
            self.logger.info(f'[L4] 闸门3 触发: 模型自评证据不足 -> 拒答 (原文: {str(raw)[:120]})')
            return QAResult.refuse(REFUSE_TEXT, stage='l4', reason='insufficient_evidence',
                                   candidates=pages_sent, cited=cited_pages)

        # 只保留真送进去的页(模型偶会写没给的页); 一个都不剩就退回全部候选页。两级身份直接比。
        cited = [key for key in cited_pages if key in pages_sent]
        if not cited:
            cited = pages_sent

        elapsed = time.time() - start
        self.logger.info(f'[L4] 生成完成, 耗时 {elapsed:.2f}s, 引用页 {cited}')

        result = QAResult.rag_answer(
            answer_text, stage='l4', candidates=pages_sent, cited=cited,
            best_score=round(best_score, 4), elapsed=round(elapsed, 3),
        )

        # ---- ③ 回填(命中才回填; 拒答不回填, 见家规 R3) ----
        # 只在无历史时回填: 缓存 key 不含历史这一维, 回填会让后续轮次读到别的上下文下的答案。
        if str(history or '').strip():
            self.logger.info('[L4] 多轮场景, 跳过缓存回填(缓存 key 不含历史)')
        else:
            self._backfill(query, answer_text, cited)

        return result

    def _backfill(self, query, answer_text, cited):
        """
        函数功能: 把命中的 L4 答案写入 Redis(TTL = TTL_L4_RAG)。
        """
        if self.redis_client is None:
            return
        try:
            key = cache_key(query)
            self.redis_client.set_data(key, {
                'answer': answer_text,
                'origin': 'l4_rag',
                'cited_pages': cited,
            }, ex=self.cache_ttl)
            self.logger.info(f'[L4] 已回填缓存 {key} (ttl={self.cache_ttl}s)')
        except Exception as e:
            # 回填是旁路, 失败不能影响已经拿到的答案
            self.logger.warning(f'[L4] 回填缓存失败(不影响回答): {e}')


if __name__ == '__main__':
    # ============================================================
    # 测试代码(仅直接运行本文件时执行): 先测不依赖模型的解析逻辑与闸门判据, 再做真实链路冒烟
    #   跑法: python rag_qa/core/multimodal_qa.py
    # ============================================================
    # ① 解析: 正常「足够」
    ok, pages, ans = parse_vl_output('证据: 足够\n页码: 31, 39\n回答: 电动尾门可通过钥匙开启。')
    _expect_pages = [(conf.MM_DEFAULT_BOOK, 31), (conf.MM_DEFAULT_BOOK, 39)]
    print(f'① 解析足够: sufficient={ok} pages={pages} answer={ans!r} '
          f'{"OK" if ok and pages == _expect_pages and "电动尾门" in ans else "FAIL"}')

    # ② 解析: 「不足」必须判为拒答
    ok, pages, ans = parse_vl_output('证据: 不足\n页码: 无\n回答: 抱歉，用户手册中没有找到相关信息，建议联系客服确认。')
    print(f'② 解析不足: sufficient={ok} {"OK" if not ok else "FAIL"}')

    # ③ 解析: 多行正文要完整保留(不能只取第一行)
    ok, pages, ans = parse_vl_output('证据: 足够\n页码: 12\n回答: 第一步, 打开设置。\n第二步, 选择座椅。')
    print(f'③ 多行正文: {ans!r} {"OK" if "第二步" in ans else "FAIL"}')

    # ④ 解析: 错误哨兵 -> 拒答
    ok, _, _ = parse_vl_output(ERROR_PREFIX + '调用视觉模型失败(超时)')
    print(f'④ 错误哨兵: sufficient={ok} {"OK" if not ok else "FAIL"}')

    # ⑤⑥ 闸门2 判据: 看"单条腿原始分", 不是 RRF 分。
    #   注入假 retriever / 假 llm, 整条链路不出网、不碰 Milvus。
    from rag_qa.core.image_leg import CandidatePage

    class _FakeRetriever:
        def __init__(self, candidates):
            self._candidates = candidates

        def retrieve(self, query=None, image=None):
            return self._candidates

    class _FakeLLM:
        def chat_vl(self, prompt, images=None, **kwargs):
            return '证据: 足够\n页码: 3\n回答: 假答案。'

    def _make_service(candidates):
        svc = MultimodalQAService(retriever=_FakeRetriever(candidates),
                                  llm=_FakeLLM(),
                                  redis_client=None,
                                  load_on_init=False)
        svc._backfill = lambda *args, **kwargs: None   # 断言只验判据, 不碰 Redis
        return svc

    # ⑤ 无阈值闸门(2026-09-29 用户裁决: 阈值拒答去掉): 有候选(哪怕零分)即放行给生成。
    #   若哪天又加回分数卡, 这条会翻车。
    zero = [CandidatePage(source_id=conf.MM_DEFAULT_BOOK, page=1, score=0.0)]
    got = _make_service(zero).answer('随便问问')
    print(f'⑤ 零分候选放行 -> source={got.source} '
          f'{"OK" if got.source == "rag" else "FAIL"}')
    assert got.source == 'rag', f'⑤ 有候选(零分)应放行生成, 实际 source={got.source} meta={got.meta}'

    # ⑥ 闸门3 模型自评不足仍拦(阈值去掉了, 这条是剩下的真闸门)。
    class _FakeLLMInsufficient:
        def chat_vl(self, prompt, images=None, **kwargs):
            return '证据: 不足\n页码: 无\n回答: 抱歉，用户手册中没有找到相关信息。'
    svc3 = MultimodalQAService(retriever=_FakeRetriever(zero),
                               llm=_FakeLLMInsufficient(),
                               redis_client=None, load_on_init=False)
    svc3._backfill = lambda *a, **k: None
    got = svc3.answer('随便问问')
    print(f'⑥ 模型自评不足 -> source={got.source} reason={got.meta.get("reason")} '
          f'{"OK" if got.source == "refuse" else "FAIL"}')
    assert got.source == 'refuse', f'⑥ 闸门3 仍应拒答, 实际 source={got.source} meta={got.meta}'

    # ⑦ 真实链路(库没建好/无权重会跳过, 不算失败)
    try:
        svc = MultimodalQAService(load_on_init=True)
        print(f'⑦ collection 现有 {svc.retriever.store.count()} 页')
        r = svc.answer('电动尾门怎么打开和关闭')
        print(f'   问「电动尾门怎么打开和关闭」-> source={r.source} hit={r.hit} '
              f'answer={r.answer[:60]!r}')
    except Exception as e:
        print(f'⑦ 真实链路跳过: {type(e).__name__}: {e}')

    