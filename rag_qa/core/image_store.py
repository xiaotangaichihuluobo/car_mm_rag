"""
image_store.py —— 图片腿的页级多向量库(Milvus 封装)。car_mm 下两张表:
  MultimodalPageStore → colqwen2_page_vectors  每页 ~747×128(patches 数组; 只被 fetch_vectors 读回)
  MultimodalFDEStore  → colqwen2_page_fde      每页 1×20480, IP(阶段1 粗筛, 生产在用)
  主键都是 `source|page`, 同进同出。

检索路径 = 「FDE 粗筛(FLAT 精确) + 候选内本地精确 MaxSim(image_leg.py numpy)」。
  旧的多向量 MAX_SIM ANN(search/search_filtered)v2.6.4 下固定错序再接淘汰——**已删, 别加回**。
两 store 都没有 release(); 放 collection 直接 `self.client.release_collection(...)`。
只配 ColQwen2, 别把 BGE-M3/Reranker 加回来。
"""

import os
import sys

# ---- 路径引导: 把项目根放进 sys.path, 让本文件既能被 import, 也能直接 python 运行 ----
# 本文件在 rag_qa/core/, 往上退 2 层到项目根。
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from pymilvus import MilvusClient, DataType

from base.config import Config
from base.logger import logger

conf = Config()

# ColQwen2 每个 patch 的向量维度, 固定 128(换 ColQwen2.5 也仍是 128)
COLQWEN2_EMB_DIM = 128


def _ensure_database(uri, database):
    """Milvus database 不存在时创建它(幂等)。模块级函数因为两个类都用。

    `MilvusClient(uri, db_name=<不存在的库>)` 构造本身不报错, 第一次 RPC 才抛 database
    not found —— 必须先建库再连(R2b); 别写"捕获构造期异常"的代码, 那条路扑空。
    """
    admin = MilvusClient(uri=uri)
    try:
        if database not in admin.list_databases():
            admin.create_database(database)
            logger.info(f"已创建 Milvus database: {database}")
    finally:
        admin.close()


class MultimodalPageStore(object):
    """
    职责: 页级多向量的建集合 / 入库 / 检索。
    """

    def __init__(self,
                 collection_name=conf.MM_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MM_MILVUS_DATABASE,
                 ensure_database=True):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger
        self.uri = f'http://{self.host}:{self.port}'

        # 库不存在先建: MilvusClient 带不存在的 db_name 会抛异常而不是帮你建。
        #   `ensure_database=False` 构造参数要留: 冒烟/自检脚本在传。
        if ensure_database:
            _ensure_database(self.uri, self.database)

        # 连接 Milvus(服务须已启动)。db_name 须本链路专用库, 别沿用文本链路的 itcast。
        self.client = MilvusClient(uri=self.uri, db_name=self.database)

    def create_collection(self, max_capacity, drop_existing=False):
        """
        建集合 + 建索引。

        :param max_capacity:  patches 数组的容量上限
                              (留余量, 取 全库最大 patch 数 * 1.2 + 16; 不够会让插入报错)
        :param drop_existing: 是否先删掉同名集合(重建库时用)
        """
        capacity = int(max_capacity)

        if drop_existing and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            self.logger.info(f"已删除旧集合 {self.collection_name}")

        # 内层 struct: 只放一个字段 emb —— 该 patch 的单条 128 维向量
        struct_schema = self.client.create_struct_field_schema()
        struct_schema.add_field("emb", DataType.FLOAT_VECTOR, dim=COLQWEN2_EMB_DIM)

        # 主键 VARCHAR id="source|page"(Milvus 无复合主键, 页号缺来源维度); source_id/page 为普通
        #   标量字段供按书过滤/删除/还原页号; auto_id=False 身份显式拼。
        schema = self.client.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=200)
        schema.add_field("source_id", DataType.VARCHAR, max_length=100)
        schema.add_field("page", DataType.INT64)
        schema.add_field("patches",
                         DataType.ARRAY,
                         element_type=DataType.STRUCT,
                         struct_schema=struct_schema,
                         max_capacity=capacity)

        self.client.create_collection(self.collection_name, schema=schema)
        self.logger.info(f"已建集合 {self.collection_name}, max_capacity={capacity}")

        # 建索引: 必须 HNSW(FLAT 不被 ArrayOfVector 接受), metric 传字符串
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="patches[emb]",
                               index_type="HNSW",
                               metric_type="MAX_SIM_COSINE",
                               params={"M": 16, "efConstruction": 64})
        self.client.create_index(self.collection_name, index_params)
        self.logger.info("HNSW 索引 (patches[emb], MAX_SIM_COSINE) 已建")

    def insert_pages(self, page_vecs, source_id, batch_size=16):
        """
        入库: {页号: (n,128) 的向量矩阵} -> 每页一个 entity, 身份 id="source|page"。

        :param page_vecs:  dict {page(int): numpy (n,128)}
        :param source_id:  这批页所属的书号(决定 id / source_id 字段, 删书也靠它)
        :param batch_size: 单次 insert 的页数。354 页一次插入的 RPC 负载太大, 分批更稳
        :return: 实际插入的页数
        """
        rows = []
        for page, vecs in sorted(page_vecs.items()):
            # pymilvus 3.0.1 行解析器不支持 struct.emb 传 list-of-lists, 必须一个 patch 一个 struct
            patch_structs = [{"emb": vec.tolist()} for vec in vecs]
            # 主键必须带书号: 库里书可能不止一本, 同页号会撞车互相顶掉
            pages_id = f"{source_id}|{int(page)}"
            rows.append({"id": pages_id,
                         "source_id": source_id,
                         "page": int(page),
                         "patches": patch_structs})

        inserted = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            self.client.insert(collection_name=self.collection_name, data=batch)
            inserted += len(batch)
            self.logger.info(f"  已入库 {inserted}/{len(rows)} 页")
        return inserted

    def load(self):
        """把集合载入内存(检索前必须)。已加载会抛异常, 不算错误。"""
        try:
            self.client.load_collection(self.collection_name)
        except Exception as e:
            self.logger.warning(f"load_collection 提示: {e}")

    def has_collection(self):
        return self.client.has_collection(self.collection_name)

    def count(self):
        """库里已有多少页(建库/自检时用来确认没建空)。"""
        if not self.has_collection():
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return stats.get('row_count', 0)

    def delete_by_source(self, source_id):
        """按书号软删该书的全部页。Milvus delete 是软删, 须 compact/有新增前确保索引覆盖。

        :param source_id: 书号
        :return: 删除行数(Milvus 返回 delete_count, 取不到时按 0 计)
        """
        if not self.has_collection():
            return 0
        expr = f'source_id == "{source_id}"'
        try:
            result = self.client.delete(collection_name=self.collection_name, filter=expr)
            return int(result.get('delete_count', 0))
        except Exception as e:
            self.logger.warning(f"delete_by_source({source_id}) 失败: {e}")
            return 0

    def fetch_vectors(self, pages=None, batch_size=50):
        """
        按身份(书+页)从集合里**复制**出这些页的全部 patch 向量(经 query(), 不是搜索)。

        多向量集合里存的就是每页一组 patch 向量(结构 patches[].emb)。用 query() 按
        主键读回, **不走 ANN 检索** —— 这是"检索回来的向量", 与建库写入/磁盘缓存
        逐点相等(建库自检验过)。阶段2 的本地精确 MaxSim 就以它为输入, 不依赖本地文件。

        :param pages:  要读的身份[(source_id, page), ...]; None=读全部页(约 135MB, 回退通路用)
        :param batch_size: 单次 query 的页数, 控制单次 gRPC 负载
        :return: {(source_id, page): (n,128) float32}
        """
        out = {}
        # 全量(pages=None): 一条不设 filter 的 query 读全部(135MB, 回退通路, 只读一次并 memo)。
        # 分页列表(pages 给定): 分批用 `id in [...]` 控制单次 gRPC 负载。
        if pages is None:
            # 空 filter 必须带 limit(Milvus 要求); 全库数百页, 给足量。
            raw = self.client.query(collection_name=self.collection_name,
                                    output_fields=["source_id", "page", "patches"],
                                    limit=2000)
            for row in raw:
                key = (row["source_id"], int(row["page"]))
                embs = [s["emb"] for s in row["patches"]]
                out[key] = np.asarray(embs, dtype=np.float32)
            return out

        targets = sorted(pages)
        for start in range(0, len(targets), batch_size):
            chunk = targets[start:start + batch_size]
            # 主键就是 id="source|page", 按它精确取回, 不跨书歧义
            id_list = ','.join(f'"{sid}|{p}"' for sid, p in chunk)
            batch_expr = 'id in [%s]' % id_list
            raw = self.client.query(collection_name=self.collection_name,
                                    filter=batch_expr,
                                    output_fields=["source_id", "page", "patches"])
            for row in raw:
                key = (row["source_id"], int(row["page"]))
                embs = [s["emb"] for s in row["patches"]]
                out[key] = np.asarray(embs, dtype=np.float32)
        return out


class MultimodalFDEStore(object):
    """
    职责: MUVERA 阶段1 的单向量库 —— 每页一条定长 FDE 向量, metric 是 IP(MIPS)。

    用 IP 而非 COSINE: MUVERA 保证建立在内积上(⟨FDE(q),FDE(d)⟩≈MaxSim), COSINE 会归一掉
    文档长度影响, 那是另一算法。默认 FLAT: 354×20480=29MB, 穷举 MIPS 毫秒级, 评估不掺 ANN 账;
    语料涨了切 HNSW(fde_index)。与 MultimodalPageStore 同库两张表、主键都是 source|page,
    **必须同进同出**(只建一张是坏库)。不走 A5: ARRAY[STRUCT] 是多向量约束, FDE 是单条
    FLOAT_VECTOR, 用普通 create_schema 即可。
    """

    def __init__(self,
                 collection_name=conf.MM_FDE_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MM_MILVUS_DATABASE,
                 index_type=conf.MM_FDE_INDEX,
                 ensure_database=True):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.index_type = index_type
        self.logger = logger
        self.uri = f'http://{self.host}:{self.port}'

        # 与 MultimodalPageStore 同一个理由: 库不存在时必须先建, 否则第一次 RPC 才炸
        if ensure_database:
            _ensure_database(self.uri, self.database)

        self.client = MilvusClient(uri=self.uri, db_name=self.database)

    def create_collection(self, dim, drop_existing=False):
        """
        建集合 + 建索引。FDE 是**单条 FLOAT_VECTOR**, 不走 A5 那套 ARRAY[STRUCT]。

        :param dim:           FDE 维度(必须等于 FDEEncoder.dim)
        :param drop_existing: 是否先删掉同名集合(重建时用)
        """
        dim = int(dim)

        if drop_existing and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            self.logger.info(f"已删除旧集合 {self.collection_name}")

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=200)
        schema.add_field("source_id", DataType.VARCHAR, max_length=100)
        schema.add_field("page", DataType.INT64)
        schema.add_field("fde", DataType.FLOAT_VECTOR, dim=dim)

        self.client.create_collection(self.collection_name, schema=schema)
        self.logger.info(f"已建集合 {self.collection_name}, dim={dim}")

        # FLAT 是穷举、不建图, 那套 HNSW 的建图参数它不认 —— 传了会被拒, 所以分开给
        if self.index_type == "HNSW":
            build_params = {"M": 16, "efConstruction": 64}
        else:
            build_params = {}

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="fde",
                               index_type=self.index_type,
                               metric_type="IP",
                               params=build_params)
        self.client.create_index(self.collection_name, index_params)
        self.logger.info(f"{self.index_type} 索引 (fde, IP) 已建")

    def insert_pages(self, page_fdes, source_id, batch_size=64):
        """
        入库: {页号: (dim,) 的 FDE} -> 每页一个 entity, 身份 id="source|page"。

        :param page_fdes:  dict {page(int): numpy (dim,)}
        :param source_id:  这批页所属的书号(决定 id / source_id 字段, 删书也靠它)
        :param batch_size: 单次 insert 的页数。单向量比多向量小得多, 可以一次多插些
        :return: 实际插入的页数
        """
        rows = []
        for page, fde in sorted(page_fdes.items()):
            rows.append({"id": f"{source_id}|{int(page)}",
                         "source_id": source_id,
                         "page": int(page),
                         "fde": np.asarray(fde, dtype=np.float32).tolist()})

        inserted = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            self.client.insert(collection_name=self.collection_name, data=batch)
            inserted += len(batch)
            self.logger.info(f"  已入库 {inserted}/{len(rows)} 页 FDE")
        return inserted

    def load(self):
        """把集合载入内存(检索前必须)。已加载会抛异常, 不算错误。"""
        try:
            self.client.load_collection(self.collection_name)
        except Exception as e:
            self.logger.warning(f"load_collection 提示: {e}")

    def has_collection(self):
        return self.client.has_collection(self.collection_name)

    def count(self):
        """库里已有多少页。"""
        if not self.has_collection():
            return 0
        return self.client.get_collection_stats(self.collection_name).get('row_count', 0)

    def search(self, fde_vec, limit):
        """
        阶段1: 一条 FDE 查询向量 -> 候选页号。

        :param fde_vec: numpy (dim,) —— FDEEncoder.encode_query() 的输出
        :param limit:   返回多少页(即 fde_topk)
        :return: [(source_id, page, score), ...] 按 score 降序; score 是 FDE 内积(≈ MaxSim,
                 但**系统性偏大**: 和里含"同桶但非最优配对"的交叉项, 别当绝对值用)
        """
        raw = self.client.search(collection_name=self.collection_name,
                                 data=[np.asarray(fde_vec, dtype=np.float32).tolist()],
                                 anns_field="fde",
                                 search_params={"metric_type": "IP", "params": {}},
                                 limit=limit,
                                 output_fields=["source_id", "page"])
        results = []
        for hit in raw[0]:
            results.append((hit["source_id"], int(hit["page"]), float(hit["distance"])))
        return results

    def delete_by_source(self, source_id):
        """按书号软删该书的全部 FDE 行。Milvus 软删, 与 MultimodalPageStore 同语义。

        :param source_id: 书号
        :return: 删除行数(取不到 delete_count 时按 0 计)
        """
        if not self.has_collection():
            return 0
        expr = f'source_id == "{source_id}"'
        try:
            result = self.client.delete(collection_name=self.collection_name, filter=expr)
            return int(result.get('delete_count', 0))
        except Exception as e:
            self.logger.warning(f"delete_by_source({source_id}) 失败: {e}")
            return 0
