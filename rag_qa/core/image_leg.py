# 召回层(PageRetriever): 单通道多模态页级检索。
#   文档侧多模态, 查询侧文本: 页库存的是**页图 patch 向量**(页级), 查询把**问题文字**
#   经 ColQwen2 编成同空间的 query patch, 走 FDE(muvera)粗排 + 本地精确 MaxSim 精排。
#   入口 retrieve(query)。图片上传不参与检索 —— 查询只有文字。

import numpy as np
import os
import sys
from dataclasses import dataclass, field

# ---- 路径引导: 把项目根放进 sys.path, 让本文件既能被 import, 也能直接 python 运行 ----
# 本文件在 rag_qa/core/, 往上退 2 层到项目根。
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from base.config import Config
from base.logger import logger
from rag_qa.core.colqwen2_encoder import get_encoder
from rag_qa.core.fde_encoder import FDEEncoder
from rag_qa.core.image_store import MultimodalPageStore

conf = Config()


@dataclass
class CandidatePage:
    """单通道检索返回的一页候选。身份是两级 (source_id, page)。

    page  = 本候选的代表页
    pages = 本候选覆盖的页集合(单通道检索是页级, 恒为该页一个)
    score = 阶段2 精确 MaxSim 原始分(拿去走拒答闸门)
    """
    source_id: str
    page: int
    score: float
    pages: list = field(default_factory=list)

    def __post_init__(self):
        if not self.pages:
            self.pages = [self.page]

    @property
    def key(self):
        """身份元组, 作 dict 键 / 序列化用。"""
        return (self.source_id, self.page)


def page_image_path(source_id, page):
    """书 + 页 -> 渲染好的 PNG 绝对路径(在源书子目录下)。"""
    return os.path.join(conf.book_image_dir(source_id), f'p{int(page):03d}.png')


# ============================================================
# 阶段2 精确 MaxSim(本地, 不走 Milvus 多向量检索)
# ============================================================
# 为什么必须本地精确(2026-09-28 实测): Milvus v2.6.4 的多向量 MAX_SIM 检索是
#   "后置过滤的劣召回 HNSW" —— ef / 建图参数都不影响它, 在 354 页(26万 patch)下
#   固定错序。即使在候选里真选已被 FDE 阶段1 列为 rank0, 它也把真选排掉(实测 5/6)。
#   数据、FDE、flush 均已逐一排除为原因 —— 精确检索(逐候选逐 patch 比对)在本地
#   numpy 实现是 6/6 全对的。因此阶段2 改为**本地精确计算标准 MaxSim**:
#   每条 query patch 取它在 doc 里最相似 patch 的相似度, 再对全部 query patch 求和。
#
#   成本为何恒定: 候选页来自 FDE 阶段1 锁定的固定几十页(FDE 数据不随语料规模涨),
#   本地只在几十页上做精确 MaxSim —— 不随全库 354 还是 3万页增长。
#
#   输入向量来源: **从 Milvus 多向量集合 query() 复制回来**(image_store.fetch_vectors),
#   不是本地 page_vecs 文件 —— 生产真源是 Milvus, 不该依赖某台机器上的缓存文件。
# ------------------------------------------------------------


def exact_maxsim(query_vecs, doc_vecs):
    """标准 MaxSim 除以 |Q|(ColBERT 查长归一): (Σ_q max_d <q,d>) / nq。

    向量已 L2 归一(ColQwen2 输出即如此), 内积即余弦。除以 query patch 数 nq 后,
    量纲是"每条 query patch 的平均最优余弦"(≈0~1), 跨 query 长短可比 ——
    供单一全局拒答阈值用; 同 query 内 nq 是常数, 排序与归一前逐位一致。
    doc 侧不用除: 对每条 query patch 已 max 掉 doc 长, 天然对文档长度免疫。
    :param query_vecs: (nq, dim) —— 查询(文字)patch 向量组
    :param doc_vecs:   (nd, dim) —— 一页(候选文档)的 patch 向量
    :return: float 分数
    """
    sim = doc_vecs @ query_vecs.T                  # (nd, nq): doc_j 与 query_i 的内积
    return float(sim.max(axis=0).sum() / sim.shape[1])  # Σ 每条 query patch 最优, ÷|Q| 查长归一


def rank_by_exact_maxsim(query_vecs, pages, k, page_vecs):
    """在给定的候选页集合里, 本地精确 MaxSim 排序, 取 top k。

    :param query_vecs: (nq, dim) 查询(文字)patch 向量
    :param pages:      候选身份[(source_id, page), ...](阶段1 FDE 给)
    :param k:          返回多少页
    :param page_vecs:  {(source_id, page): (n,128) 向量} —— 由 image_store.fetch_vectors 从 Milvus 取回
    :return: [(source_id, page, MaxSim 分数), ...] 按分数降序; 同分按身份升序, 保证可复现
    """
    scored = []
    for key in pages:
        dv = page_vecs.get(key)
        if dv is None or len(dv) == 0:
            continue
        source_id, page = key
        scored.append((source_id, int(page), exact_maxsim(query_vecs, dv)))
    scored.sort(key=lambda t: (-t[2], t[0], t[1]))
    return scored[:k]


class PageRetriever(object):
    """
    职责: 给定「问题文字」, 返回候选页 topN。
    这一层只负责"召回", 不判断该不该拒答、也不生成 —— 那是 multimodal_qa 的事。
    """

    def __init__(self, store=None, encoder=None, load_on_init=True, fde_store=None):
        self.logger = logger
        self.leg_topk = conf.MM_LEG_TOPK     # 阶段2 返回的页数
        self.fde_topk = conf.MM_FDE_TOPK     # 阶段1 muvera 粗筛候选上限

        self.store = store or MultimodalPageStore()

        # FDE 库惰性: 首次用到才连(不用的场景不必多开一次链接)
        self._fde_store = fde_store

        # FdeEncoder 只存参数; ColQwen2 只解析路径 —— 权重到首次编码才进显存
        self.fde_encoder = FDEEncoder()
        self.encoder = encoder or get_encoder()

        if load_on_init:
            self.store.load()

    @property
    def fde_store(self):
        """MUVERA 阶段1 的单向量库(页级 FDE)。首次访问才连。"""
        if self._fde_store is None:
            from rag_qa.core.image_store import MultimodalFDEStore
            self._fde_store = MultimodalFDEStore()
        return self._fde_store

    def image_leg(self, query, k=None):
        """
        单通道检索: 文字 query -> patch -> FDE(muvera 粗排) -> 本地精确 MaxSim 精排 -> topK 页。

        阶段1 在页级 FDE 库上做 muvera MIPS 粗排, 取 fde_topk 候选页;
        阶段2 只在这批候选里, 从 Milvus 取回页 patch 向量, 本地精确 MaxSim 排序取 top k。

        :param query: 问题文字
        :param k: 返回多少页(默认 leg_topk)
        :return: [(source_id, page, MaxSim 分数), ...] 降序
        """
        if k is None:
            k = self.leg_topk
        if not query or not str(query).strip():
            return []

        vectors = self.encoder.encode_text([str(query)])[0]
        query_fde = self.fde_encoder.encode_query(vectors)

        candidates = self.fde_store.search(query_fde, limit=self.fde_topk)
        if not candidates:
            # 阶段1 一无所获, 不必再花一次 RPC —— 上层会走闸门1 (无候选 -> 拒答)
            self.logger.warning('MUVERA 阶段1 无候选, 返回空')
            return []

        pages = [(sid, page) for sid, page, _score in candidates]
        pv = self.store.fetch_vectors(pages)
        self.logger.info(f'MUVERA 阶段1 召回 {len(pages)} 页 -> 本地精确 MaxSim 取 top{k}')
        return rank_by_exact_maxsim(vectors, pages, k, pv)

    def retrieve(self, query=None, image=None):
        """
        [L4] 召回: 文字 query -> 多模态页库(muvera+maxsim) -> 候选页。

        :param query: 问题文字
        :param image: 预留参数, 忽略 —— 查询只有文字, 图片不参与检索
        :return: [CandidatePage, ...] 按 MaxSim 分降序, 长度 <= leg_topk
        """
        if image is not None:
            self.logger.info('已忽略传入图片: 本系统查询只有文字')
        hits = self.image_leg(query)
        return [CandidatePage(source_id=sid, page=page, score=score, pages=[page])
                for sid, page, score in hits]


if __name__ == '__main__':
    # ============================================================
    # 自检(仅直接运行本文件时执行)
    #   跑法: python rag_qa/core/image_leg.py
    #   ①②③⑤ 是纯逻辑断言, 不需要模型/数据库, 永远能跑;
    #   ⑧ 需要 Milvus + 页库已建 + 本地 ColQwen2 权重; 缺则跳过。
    # ------------------------------------------------------------
    # ① exact_maxsim 数学
    _q = np.array([[1.0, 0.0], [0.0, 1.0]])     # 两条正交 query patch
    _doc = np.array([[1.0, 0.0], [0.9, 0.1]])
    # query[0]=[1,0] 在所有 doc 里最大=1.0; query[1]=[0,1] 与 doc[1]=[0.9,0.1] 最大=0.1
    #   -> 和 = 1.0 + 0.1 = 1.1; 除以 |Q|=2 查长归一 -> 0.55
    _ms = exact_maxsim(_q, _doc)
    _ok = abs(_ms - 0.55) < 1e-6
    print(f'① exact_maxsim 正交/相似(查长归一): {_ms:.4f} (期望 0.55) {"OK" if _ok else "FAIL"}')
    assert _ok

    # ② CandidatePage 默认页集合 = 代表页
    _c = CandidatePage(source_id='train_a', page=31, score=1.2)
    _ok = (_c.pages == [31] and _c.key == ('train_a', 31))
    print(f'② CandidatePage 默认 pages=[31], key=("train_a",31) {"OK" if _ok else "FAIL"}')
    assert _ok

    # ③ page_image_path 拼接
    _path = page_image_path('train_a', 31)
    _ok = _path.replace('\\', '/').endswith('train_a/p031.png')
    print(f'③ page_image_path -> {_path} {"OK" if _ok else "FAIL"}')
    assert _ok

    # ⑤ 空 query 必须返回空(零依赖)。走的是空 query 提前 return, 不碰任何依赖。
    _dummy = object()
    _bare = PageRetriever(store=_dummy, encoder=_dummy, load_on_init=False)
    assert _bare.image_leg('') == [], '空 query 应返回 []'
    assert _bare.image_leg(None) == [], 'None query 应返回 []'
    print('⑤ 空 query 返回 [] OK')

    # ⑧ 真实检索(需要 Milvus + 页库 + FDE 库 + 本地 ColQwen2 权重; 缺则跳过)
    #   判据: 文字 query 必须命中它描述的页 —— 尾门=~31, 座椅=~50。
    #   这是"文字走 muvera+maxsim"的端到端探测器, A1 空间对齐的正式回归项。
    #   try 只包住"起依赖"; 断言必须在 try 外面(真回归不能被宽 except 吞成"跳过")。
    from rag_qa.core.image_store import MultimodalFDEStore

    _mm_ready = None
    try:
        _mm_store = MultimodalPageStore()
        _mm_fde = MultimodalFDEStore()
        if _mm_store.count() and _mm_fde.count():
            _mm_ready = PageRetriever(store=_mm_store, fde_store=_mm_fde, load_on_init=True)
        else:
            print(f'⑧ MUVERA 两阶段跳过: 库是空的(多向量 {_mm_store.count()} 页 / '
                  f'FDE {_mm_fde.count()} 页)—— 先跑 build_multimodal_kb.py 和 build_fde_kb.py')
    except Exception as e:
        print(f'⑧ MUVERA 两阶段跳过: {type(e).__name__}: {e}')

    if _mm_ready is not None:
        _cases = [('车外如何打开和关闭电动尾门', {30, 31}),
                  ('驾驶员座椅如何电动调节', {50, 51})]
        for _q_text, _gold in _cases:
            _hits = _mm_ready.image_leg(_q_text, k=5)
            _top = [p for _sid, p, _s in _hits]
            _hit = bool(set(_top) & _gold)
            print(f'⑧ 「{_q_text}」-> top5 {_top}(金页 {_gold}) -> {"HIT" if _hit else "MISS"}')
            assert _hit, f'⑧ 文字 query 没命中金页 {_gold}: {_top}'
        print('⑧ 单通道文字检索 OK')