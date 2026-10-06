# carRAG · 汽车售后智能问答系统

统一管理汽车售后问答三级链路(Redis 直答 → MySQL/BM25 检索答 → 多模态 RAG 生成答),共享一套配置(MySQL / Redis / Milvus / 千问 DashScope)。全部数据依赖由一个 docker 编排(多容器、官方镜像)拉起,应用在宿主机跑。

- `base/` — 统一配置管理(config.ini + .env)
- `mysql_qa/` — MySQL + BM25 检索 + Redis 缓存问答链路
- `rag_qa/` — RAG 问答链路(页渲染、页级视觉检索、生成)
- `scripts/` — 建库 / 评估 / 初始化脚本
- 语料 — 领克手册 `rag_qa/data/car_data/`(训练 PDF);汽车售后 QA `mysql_qa/data/汽车售后问答.csv`
- `docker-compose.yml` — mysql / redis / etcd / minio / milvus 五个数据容器(端口见下)

> 数据准备与 MySQL/Redis 链路见各子模块 README;本文档正文为**多模态问答链路最终方案(定稿)**。

---

## 启动与数据初始化

**环境前置**(宿主机,容器只装数据不装应用层):运行用 `EduRAG` conda env;本地模型权重在 `rag_qa/models/colqwen2_local/`(勿删)。

### 1) 拉起五个数据容器

```bash
docker compose up -d
```

起 `mysql / redis / etcd / minio / milvus` 五个服务(官方镜像 + 命名卷持久化),连等 `healthy`。端口对齐 `config.ini`:

| 服务 | host:container | 说明 |
|---|---|---|
| MySQL | `13306:3306` | 避开常见 3306 |
| Redis | `16379:6379` | 避开常见 6379 |
| Milvus | `19538:19530` | 避开常见 19530 与 research-agent 的 19531 |
| etcd / minio | 容器内不对外 | 仅 compose 网络内部 |

### 2) 数据初始化(独立脚本,只跑一次;已有数据可跳过)

```bash
# MySQL:建表 jpkb + 从 mysql_qa/data/汽车售后问答.csv 灌 9755 条
python mysql_qa/db/mysql_client.py --force

# Milvus 多模态检索:页向量建库 → FDE 阶段1 建库(顺序不能反,GPU 建库走多模态脚本)
python scripts/build_multimodal_kb.py
python scripts/build_fde_kb.py
```

> 注:空库 `subjects_kg` 由 mysql 容器首次启动时自动执行 `scripts/init_db.sql` 建好;上面脚本只负责装数据。`python -c "from base.config import Config; print(Config().MYSQL_PORT)"` 可抽查配置端口。

### 3) 起应用

```bash
python app.py   # FastAPI → http://localhost:8000
```

**自检**:`docker compose ps` 全 healthy;`SELECT COUNT(*) FROM subjects_kg.jpkb` 应 9755;Milvus 集合各 354 页;`curl http://localhost:8000/health`(若有)应 200。

**已初始化/快速路径**:数据已在 → 只执行 1 + 3 即可直接问答。

---

## 多模态问答链路 — 最终方案(定稿)

### 一句话选型

**检索层用「ColPali 系整页视觉检索」:PDF 每页渲染成一张图,由 ColQwen2 直接编码成多向量,原文与图同空间检索,不做 OCR、不做 caption;生成层用千问 qwen-vl 看图作答。** 文字问题同样编成与页图同空间的多向量,单通道检出整页,不另设文字腿。

### 为什么必须整页视觉检索(数据背景)

- 当前语料**天池领克手册 `rag_qa/data/car_data/train_a.pdf`(354 页)**文字图层**完好**(145074 字符 / 私用区 0)→ 文本抽取可行,**整页视觉检索仍是主路径**,因为答案证据与页面图绑定。
- 问答中图片/图表是**答案的核心证据**(座椅怎么调、尾门怎么开,答案和页面图绑定),系统必须能把「具体的页/图」召回,而不是只召回一段孤零零的文字。

---

### 最终架构(单通道)

```
                 建库(一次性)                                  查询(每次)
 ┌─────────────────────────────┐        ┌──────────────────────────────────────────┐
 │ PDF ──每页渲染──► 页面图 ────│        │ 文字问题 ──► ColQwen2 encode_text ─► 多向量  │
 │     │                       │        │        (与页图同空间)                        │
 │     ▼                       │        │                  │                        │
 │ ColQwen2 视觉编码            │        │                  ▼                        │
 │  └─► 每页 ~700×128 patch 多向量│        │  阶段1 muvera FDE MIPS 粗排 → top fde_topk       │
 │     │                        │        │                  │                        │
 │     ▼                        │        │                  ▼                        │
 │ Milvus 页向量 ────────────►  │        │  阶段2 本地精确 MaxSim 精排 → top leg_topk      │
 │   + FDE 阶段1 定长向量        │        │                  │                        │
 └─────────────────────────────┘        │                  ▼                        │
                                        │          整页候选页(含页图)                  │
                                        │                  │                        │
                                        │                  ▼                        │
                                        │      qwen-vl(问题文字 + 候选页图) ──► 回答    │
                                        └──────────────────────────────────────────┘
```

**核心分工**:检索层里页面是「检索键」,生成层里页面图是「上下文」。文字 query 不另走文本向量,而是编成与页图同空间的多向量(Zip)后与页向量做 MaxSim —— **单通道**,无文字腿、无图片上传分流、无 RRF 融合。

### 关键实现点(均已实测)

| 环节 | 定稿做法 | 要点 / 坑 |
|---|---|---|
| 页渲染 | `pymupdf` 按页渲染 PNG | 文字层完好;渲染成图是为了把页当检索键 |
| 编码模型 | **ColQwen2-2B** `vidore/colqwen2-v1.0-hf` | transformers **原生 `ColQwen2ForRetrieval`**,无需 colpali-engine;`4bit nf4` 加载,峰值显存 **1.8GB**(4GB 卡稳) |
| 页向量 | 每页 ≈ **700+ 条 128 维 patch 多向量** | 编码输出 bf16,落 numpy 前先 `.float()` |
| 存储 | Milvus 2.6+ 单 collection,一页一 entity | 多向量**不能**放顶层 array-of-vector(服务器拒);必须嵌 `ARRAY[STRUCT]`,每个 patch 一个 struct、`struct.emb` 放该 patch 的单条 128 维向量 |
| 索引/检索 | `patches[emb]` HNSW,**metric 字符串 `"MAX_SIM_COSINE"`**,查询向量包成 `EmbeddingList` | pymilvus 的 MetricType 枚举**没有** MAX_SIM*,`IP/COSINE/L2` 对 ArrayOfVector 全被拒;FLAT 索引不被接受 |
| 两阶段检索 | **阶段1**:文字 query 经 `encode_text` 压成多向量,再压成 FDE 定长向量,对页级 FDE 集合 muvera MIPS 粗排 → `fde_topk`;阶段2 只在这批候选里做本地精确 `MaxSim` → `leg_topk` | FDE 四参数(seed / k_sim / dim_proj / reps)必须与建库时一致,否则不在同一空间(不报错但结果全错) |
| 生成 | qwen-vl(DashScope)输入 = 问题文字 + 候选页图 | 取候选页 top5 层给 LLM,多页让 LLM 自选证据 |

---

### 冒烟验证结果(本机:RTX 3050 Ti 4GB)

回溯记录(探针脚本已随评估期收口删除;下表为当时的跑通记录):

| # | 冒烟项 | 结果 |
|---|---|---|
| 1 | 模型 4bit 能加载并编码页 | **PASS** — 加载 11s,编码 10 页 9s,峰值显存 1.80GB;每页 ~747×128 向量 |
| 2 | 纯内存版检索(未接 Milvus) | 以图搜图 **10/10**;文本查询 **3/4** |
| 3 | Milvus Struct-Array 入库 + MAX_SIM | **PASS** — collection `colqwen2_pages`;以图搜图 **10/10**、文本 **3/4** |
| 4 | 两阶段 muvera 粗排 + MaxSim 精排 | **PASS** — 粗排在页级 FDE 库上 MIPS,精排本地精确 MaxSim,毫秒级 |

**环境备忘(踩过坑,勿重试)**:

- GPU 仅 4GB → 3B 级模型要 **int4 NF4**(bitsandbytes,仅 GPU);int8 ~3.5GB 太紧,bf16 放不下。ColQwen 无现成 GGUF/GPTQ 档,想量化只有 GPU bitsandbytes。
- transformers 锁 **4.x**(现 4.57.6);曾误升 5.x 导致 sentence-transformers(要求 <5)与 Qwen/ColQwen 系列全挂。
- **网络**:本机 huggingface_hub 连 `hf-mirror.com` HEAD 会失败,huggingface.co 被墙 → 模型**本地化**(`rag_qa/models/colqwen2_local`,勿重下)或走 **ModelScope**(同 id `vidore/colqwen2-v1.0-hf`)。

---

### 评估

当前评估把检索与生成分开跑,全部绑同一份页绑定评估数据(`eval_set`,gold 取自唯一源页、不靠答案文字定位;n=39 在域单页 + 8 域外):

| 环节 | 脚本 | 指标 |
|---|---|---|
| 检索(纯本地) | `python scripts/eval_rag.py --only retrieval` | Recall@1 53.8% / Recall@5 92.3% / MRR 0.671(池深恒 top5, n=39) |
| 生成 RAGAS(付费) | `scripts/gen_ragas_dataset.py` → `scripts/ragas_eval.py` | 上下文召回率 0.949 / 回答正确性 0.935;完整集 忠实度 0.769 / 答案相关性 0.784 / 上下文精确率 0.795 |

数据落到 `rag_qa/data/eval/`(被 gitignore,不入库)。检索段不出网、纯本地;RAGAS 段用 `qwen-plus` 打指定指标(仅该段收费)。

### 升级路径(换更大模型)

冒烟以 **ColQwen2-2B** 为可跑基线。若日后换更强 GPU(≥6GB),可平滑升 **ColQwen2.5**(Qwen2.5-VL-3B 底座,中文更强):编码维度、Collection 拓扑、检索代码**全不变**,只换模型权重与本地路径,需重新建库。以图搜图对「真实现场照片 vs 手册截图」的跨分布落差,是该方法与真机照片检索间的固有边界,留待真实数据评估。