# 把一组变长 patch 向量压成一条定长 FDE 向量, 供 MUVERA 阶段1 粗筛。
# 底层是 pymuvera(Google graph-mining MUVERA 的移植), 本文件只编排参数与入口。
#
# 两个必须做对、做错即静默失效的点:
#   ① 进 FDE 前必须 L2 归一化(库用 MAX_SIM_COSINE)。归一化收在 encode 内部、
#      不交给调用方, 建库与查询就没有各写一遍、写歪的机会。
#   ② 查询侧与建库侧必须共用同一组 H_r/P_r —— 由固定 seed 保证。
#
# 关键不变量: 查询侧桶内**求和**、文档侧桶内**求平均**。两边都求和会让
# "一个 patch 彼此相似"的页因桶聚合被撑大而虚高。
# 所以入口拆成 encode_query / encode_document 且不留单方法入口, 避免把文档当查询编码。

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from base.config import Config
from rag_qa.core.image_store import COLQWEN2_EMB_DIM

conf = Config()


class FDEEncoder(object):
    """把一组 patch 向量 (n,128) 压成一条定长 FDE 向量 (D,)。
    只处理图片腿的 patch 向量, 文字腿走 BGE-M3。"""

    def __init__(self, k_sim=None, dim_proj=None, r_reps=None, seed=None):
        self.k_sim = int(conf.MM_FDE_K_SIM if k_sim is None else k_sim)
        self.dim_proj = int(conf.MM_FDE_DIM_PROJ if dim_proj is None else dim_proj)
        self.r_reps = int(conf.MM_FDE_REPS if r_reps is None else r_reps)
        self.seed = int(conf.MM_FDE_SEED if seed is None else seed)

        # 底层编码器惰性构造(构造它不能碰重资源)
        self._encoder = None

    @property
    def dim(self):
        """FDE 维度 D = r_reps × 2^k_sim × dim_proj(默认 20×64×16 = 20480)。"""
        return self.r_reps * (2 ** self.k_sim) * self.dim_proj

    def _ensure_encoder(self):
        """惰性构造底层编码器(幂等)。"""
        if self._encoder is not None:
            return

        from pymuvera import MUVERAEncoder
        from pymuvera.config import ProjectionType

        # 必须 AMS_SKETCH: 只有它让 projection_dimension 生效, 默认档直接返回
        # dimension(=128), 会使 D = reps×2^k×128 = 163840, 超 Milvus
        # FLOAT_VECTOR 上限 32768, 建库在服务器层炸。
        self._encoder = MUVERAEncoder(
            dimension=COLQWEN2_EMB_DIM,
            num_simhash_projections=self.k_sim,
            num_repetitions=self.r_reps,
            seed=self.seed,
            projection_type=ProjectionType.AMS_SKETCH,
            projection_dimension=self.dim_proj,
        )

        # 本地公式与库自报必须一致, 否则建库与查询落在不同维度(静默失效, 见文件头 ②)
        actual = self._encoder.fde_dimension
        if actual != self.dim:
            raise ValueError(
                f'FDE 维度不一致: 本地公式算出 {self.dim}, pymuvera 算出 {actual}。'
                f'检查 k_sim / dim_proj / r_reps 的映射。')

    @staticmethod
    def _normalize(patch_vecs):
        """L2 归一化, 零范数行(全零 patch)防除零。

        pymuvera 的 prepare_embeddings() 只校验形状不归一化; 归一化收在这一处,
        建库与查询就没有各写一遍、写歪的机会。
        """
        vecs = np.asarray(patch_vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode_query(self, patch_vecs):
        """查询侧编码: 桶内**求和**(不除以桶内条数)。返回 numpy (dim,) float32。"""
        self._ensure_encoder()
        return self._encoder.encode_query(self._normalize(patch_vecs))

    def encode_document(self, patch_vecs):
        """文档侧编码: 桶内**求平均**(除以桶内条数)。

        与 encode_query 的不对称是 MUVERA 的一部分 —— 正是它让两条 FDE 的内积
        逼近 Σ_q max_d ⟨q,d⟩(Chamfer/MaxSim), 而不是"同桶所有点对相似度之和"。
        """
        self._ensure_encoder()
        return self._encoder.encode_document(self._normalize(patch_vecs))


if __name__ == '__main__':
    # 自检(零依赖, 不需 Milvus/模型/权重): 验的是 pymuvera 的**用法** ——
    # 参数映射(⑦a)、两侧语义(⑦b)、归一化(⑦c)、seed(⑦d)、与穷举 MaxSim 保序(⑦e)。
    _rng = np.random.default_rng(0)

    _enc = FDEEncoder(k_sim=4, dim_proj=8, r_reps=3, seed=42)
    _x = _rng.standard_normal((7, 128)).astype(np.float32)
    _expect_dim = 3 * (2 ** 4) * 8
    _fq_x = _enc.encode_query(_x)
    _fd_x = _enc.encode_document(_x)

    # ⑦a 维度: 本地公式 == 期望值, 两侧输出同形状
    _ok = (_enc.dim == _expect_dim
           and _fq_x.shape == (_expect_dim,)
           and _fd_x.shape == (_expect_dim,))
    print(f'⑦a FDE 维度: {_enc.dim} 期望 {_expect_dim} {"OK" if _ok else "FAIL"}')
    assert _ok, 'FDE 维度/形状不对'

    # ⑦b 两侧聚合语义必须不同: 同一向量重复 n 次必落同一桶,
    #    查询侧求和 => n 倍; 文档侧求平均 => 与 1 份完全相同。写成一样即自匹配分虚高。
    _v = _rng.standard_normal((1, 128)).astype(np.float32)
    _v3 = np.repeat(_v, 3, axis=0)
    _q1, _q3 = _enc.encode_query(_v), _enc.encode_query(_v3)
    _d1, _d3 = _enc.encode_document(_v), _enc.encode_document(_v3)
    _ok_sum = np.allclose(_q3, 3.0 * _q1, rtol=1e-4, atol=1e-4)
    _ok_avg = np.allclose(_d3, _d1, rtol=1e-4, atol=1e-4)
    print(f'⑦b 两侧聚合: query([x]*3)==3*query([x])={_ok_sum}; '
          f'doc([x]*3)==doc([x])={_ok_avg} {"OK" if _ok_sum and _ok_avg else "FAIL"}')
    assert _ok_sum, '查询侧不是桶内求和'
    assert _ok_avg, '文档侧不是桶内求平均'

    # ⑦c 归一化收在两侧内部: 整体缩放输入不该改变输出
    _ok = (np.allclose(_enc.encode_query(_x * 5.0), _fq_x, rtol=1e-4, atol=1e-4)
           and np.allclose(_enc.encode_document(_x * 5.0), _fd_x, rtol=1e-4, atol=1e-4))
    print(f'⑦c 内部归一化: encode(5X) == encode(X) 两侧 {"OK" if _ok else "FAIL"}')
    assert _ok, '没做内部 L2 归一化, 或归一化写错了'

    # ⑦d 确定性: 同 seed 逐位相同, 异 seed 必须不同 —— 守"两侧共用同一组矩阵"这条不变量
    _again = FDEEncoder(k_sim=4, dim_proj=8, r_reps=3, seed=42).encode_document(_x)
    _other = FDEEncoder(k_sim=4, dim_proj=8, r_reps=3, seed=43).encode_document(_x)
    _same = np.array_equal(_again, _fd_x)
    _diff = not np.array_equal(_other, _fd_x)
    print(f'⑦d 确定性: 同 seed 相同={_same}, 异 seed 不同={_diff} '
          f'{"OK" if _same and _diff else "FAIL"}')
    assert _same, 'seed 没接上'
    assert _diff, '异 seed 结果相同'

    # ⑦e 保序性(跨两侧): FDE 内积排序必须与穷举 MaxSim 同向
    _q = _rng.standard_normal((12, 128)).astype(np.float32)
    _doc_ok = _q + 0.1 * _rng.standard_normal((12, 128)).astype(np.float32)
    _doc_bad = _rng.standard_normal((12, 128)).astype(np.float32) * 5.0 + 20.0

    def _maxsim(query, doc):
        """两边各自 L2 归一化后, 逐行取最大相似度再求和 —— 与 MAX_SIM_COSINE 同一几何。"""
        _qn = query / np.linalg.norm(query, axis=1, keepdims=True)
        _dn = doc / np.linalg.norm(doc, axis=1, keepdims=True)
        return float((_qn @ _dn.T).max(axis=1).sum())

    _ms_ok = _maxsim(_q, _doc_ok)
    _ms_bad = _maxsim(_q, _doc_bad)
    _fq = _enc.encode_query(_q)
    _ok = ((_ms_ok > _ms_bad)
           == (_fq @ _enc.encode_document(_doc_ok) > _fq @ _enc.encode_document(_doc_bad)))
    print(f'⑦e 保序性: MaxSim {_ms_ok:.2f} vs {_ms_bad:.2f} '
          f'{"OK" if _ok else "FAIL"}')
    assert _ok, 'FDE 内积与 MaxSim 不同向'