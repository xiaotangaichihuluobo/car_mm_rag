# -*- coding: utf-8 -*-
"""
eval_rag.py —— 本项目的 RAG 评估台。

【量什么】
  检索侧  L4 单通道多模态(文字 query -> 页) / L2 题库  -> Recall@k, MRR
  分类侧  L3 意图识别                        -> 准确率, 混淆矩阵
  拒答    L4 三道闸门                        -> 混淆矩阵 + 闸门归因
  生成侧  L4 答案质量                        -> RAGAS 四指标
  端到端  编排器全链路                       -> 引用页命中率 + 延迟

【为什么是单文件】
  检索侧**纯本地免费**, 生成侧要跑 RAGAS(每题约 12 次 API 调用 = 11 次裁判 + 1 次答案生成;
  其中 context_precision 一项就占 5 次, 与召回页数(leg_topk)同阶, 改召回数这个数就变)。
  ⚠️ **"每题约 12 次"是未实测的估计**(设计文档 §7 另写"每题约 5 次", 两者互相矛盾且都无埋点
  支撑)。改代码时别把这个数当实测引用; 要真用它先补计数。成本差一个量级 → 必须能分开跑
  (--only), 调检索时才敢反复执行。

【跑法】
    cd "C:\\Users\\11384\\Desktop\\黑马学习\\阶段五\\05_EduRAG 项目\\04-代码\\integrated_qa_system"
    PYTHONPATH=. "C:\\Users\\11384\\anaconda3\\envs\\EduRAG\\python.exe" scripts/eval_rag.py --selfcheck
    PYTHONPATH=. "C:\\Users\\11384\\anaconda3\\envs\\EduRAG\\python.exe" scripts/eval_rag.py --build-set
    PYTHONPATH=. "C:\\Users\\11384\\anaconda3\\envs\\EduRAG\\python.exe" scripts/eval_rag.py --only retrieval
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from base.config import Config                                        # noqa: E402

conf = Config()

EVAL_DIR = os.path.join(_PROJECT_ROOT, 'rag_qa', 'data', 'eval')
EVAL_SET = os.path.join(EVAL_DIR, 'eval_set.json')
# 新评估源 = car_intent_5000.json 的「专业咨询」题(每条带 `page` = 生成该问的那一页)——
#    gold 干净、单页、确定, 不靠 result.json/test_question.json(旧坏源, 已删)。
INTENT_DATA = os.path.join(_PROJECT_ROOT, 'rag_qa', 'classify_data', 'car_intent_5000.json')

# gold 页超过这个数的题, Recall@5 几乎必然 = 1, 退化成噪声 -> 排除出平均分
GOLD_MAX_PAGES = 5

import random                                                         # noqa: E402

# 评估台真值**暂单书**(用户裁决: 先迁脚本, 真值重锚延后)—— gold_pages/gold_page 仍是
#   eval_set*.json 里的 int 页号, 全部属于 SOURCE_ID(默认书)。检索/生成层已迁成两级身份
#   (source_id, page), 所以在"进入评估指标"的边界上, 把检索身份投影回 SOURCE_ID 的 int 页号,
#   与 int gold 比对。别书命中多书后才有: 那时 gold 必先重锚带书号, 而目前 int gold 无法
#   与别书预比, 直接剔除 —— 这正是"真值单书"该有的语义, 不是静默丢数据。
SOURCE_ID = conf.MM_DEFAULT_BOOK


def page_to_int(ident):
    """
    把各类两级身份投影成"真值单书 int 页号", 供与 gold_pages(int) 交集/排序。

    输入形态(生产路径真实返回):
      - int                       -> 原样(单书 legacy, 兼容旧真值)
      - (source_id, page, score)  -> image_leg / generation meta 候选
      - CandidatePage             -> retrieve() 候选
      - "source|page" 引用串      -> new_main 序列化后的 cited_pages
      - "page" 纯数字串           -> 旧形(单书兜底)

    :return: int 页号(仅当该身份属于 SOURCE_ID; 别书 / 解析失败返回 None)
    """
    if isinstance(ident, bool):
        return None
    if isinstance(ident, int):
        return int(ident)
    if isinstance(ident, str):
        s = ident.strip()
        if '|' in s:
            sid, _, p = s.partition('|')
            if sid.strip() != SOURCE_ID:
                return None
            return page_to_int(p)          # 递归解析页号部分(数字串或 int)
        try:
            return int(s)                   # 纯数字串 -> 单书默认书页
        except ValueError:
            return None
    if isinstance(ident, tuple):
        sid, page = ident[0], ident[1]
    else:                             # CandidatePage
        sid, page = ident.source_id, ident.page
    if sid != SOURCE_ID:
        return None
    return page_to_int(page)


def ranked_to_ints(ranked):
    """
    把一条检索命中的身份列表映成"真值单书 int 页号"列表(剔除别书/解析失败)。

    单通道检索每条命中恒为一页(CandidatePage), 不展开跨页 span。
    """
    out = []
    for h in ranked:
        n = page_to_int(h)
        if n is not None:
            out.append(n)
    return out


def norm(text):
    """归一化: 去掉所有空白。PDF 抽出的字与摘录的空格/换行对不齐, 不比这个就没法匹配。"""
    return re.sub(r'[\s　]+', '', text or '')


_PAGE_TEXT_CACHE = None


def page_text_map():
    """354 页文字层: {页号: 该页文字}。首次调用时抽一次并缓存在内存里。"""
    global _PAGE_TEXT_CACHE
    if _PAGE_TEXT_CACHE is None:
        import fitz
        doc = fitz.open(conf.MM_PDF_PATH)
        try:
            _PAGE_TEXT_CACHE = {i: doc[i].get_text('text') for i in range(doc.page_count)}
        finally:
            doc.close()
    return _PAGE_TEXT_CACHE


def build_eval_set():
    """
    从 car_intent_5000.json 建**页绑定**评估集, 落盘到 rag_qa/data/eval/eval_set.json。

    数据源 = car_intent_5000.json 的「专业咨询」题 —— 每条带 `page`(生成该问的那个手册页),
    所以 gold **干净、单页、确定**, 不靠"答案文字在全文定位"(旧法靠 result.json 的 answer_5
    反查文字层, 空/超页漫天: 旧 203 条里 105 条空、27 条超 5 页, 仅 35% 可用)。eval_refuse
    已用同样的 JSONL+label 读法(见 INTENT_DATA), 此处照抄判据。

    组装:
      ① 「专业咨询」+ 带 page 的题按页聚成 {page: [题,...]}(2500 条覆盖 292 页 16..353)。
      ② 分层抽样摊满全册: 按页序每 CC 页取 1 页, 每页定第 1 题(确定性, 重跑稳定), K≈60。
      ③ 另抽 8 条「通用知识」域外题(gold_pages=[]) 供 refuse/e2e 用。
      ④ 写 eval_set.json: {id, question, gold_pages:[page], reference:'', source:'car_intent_5000'}。

    【reference 故意留空】页绑定题没有参考答案。生成类评估(RAGAS, 花钱)按 reference 过滤后
      为空、自然早退 —— 本次评估范围=检索(Recall@k)+拒答+端到端; 生成侧留待将来补 reference
      再跑。这是"先要干净检索数"的显式取舍, 不是静默降级。
    """
    os.makedirs(EVAL_DIR, exist_ok=True)

    # ---- ① 读 car_intent_5000(JSONL), 判据照 eval_refuse ----
    in_domain = {}          # page -> [查询, ...]
    out_domain = []
    with open(INTENT_DATA, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get('label') == '专业咨询' and item.get('page') is not None:
                in_domain.setdefault(int(item['page']), []).append(item['query'])
            elif item.get('label') == '通用知识':
                out_domain.append(item['query'])

    pages_sorted = sorted(in_domain)
    n_in = sum(len(v) for v in in_domain.values())
    print(f'car_intent_5000: 专业咨询 {n_in} 条, 覆盖 {len(pages_sorted)} 页 '
          f'({pages_sorted[0]}..{pages_sorted[-1]}); 通用知识 {len(out_domain)} 条')

    # ---- ② 分层抽样: 每 step 页取 1 页, 每页定第 1 题(确定性) ----
    K = 60
    step = max(1, len(pages_sorted) // K)
    picked_pages = pages_sorted[::step][:K]

    rows = []
    for k, page in enumerate(picked_pages, start=1):
        rows.append({
            'id': f'q{k:03d}',
            'question': in_domain[page][0],
            'gold_pages': [page],
            'reference': '',
            'source': 'car_intent_5000',
        })
    print(f'页绑定题: {len(rows)} 条, 每页 1 题, gold 单页')

    # ---- ③ 域外题(gold=[]) 供 refuse/e2e ----
    random.seed(20260916)
    for j, q in enumerate(random.sample(out_domain, 8), start=1):
        rows.append({
            'id': f'neg{j:02d}',
            'question': q,
            'gold_pages': [],
            'reference': '',
            'source': 'car_intent_5000_neg',
        })
    print(f'域外题: 补入 8 条(gold_pages=[])')

    with open(EVAL_SET, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f'-> {EVAL_SET}  共 {len(rows)} 条')

    print_spotcheck(rows, page_text_map())
    return 0


def print_spotcheck(rows, pages, n=12, seed=20260916):
    """
    抽 12 条打印出来给人眼确认。

    【为什么必须有这一步】数据集错了后面所有指标全错, 而且错得很像真的 ——
    这是"数字能翻到证据"的唯一保证。
    """
    pool = [r for r in rows if r['gold_pages']]
    random.seed(seed)
    picked = random.sample(pool, min(n, len(pool)))
    print(f'\n{"=" * 70}\n抽检 {len(picked)} 条(有 gold 的题里随机取, seed={seed})\n{"=" * 70}')
    for r in picked:
        head = norm(pages[r['gold_pages'][0]])[:40]
        print(f'[{r["id"]}] {r["question"]}')
        print(f'    gold 页: {r["gold_pages"]}')
        print(f'    首 gold 页开头: {head}')
        print()


# 单通道多模态检索精排取 leg_topk 条, 才够算 Recall@leg_topk。
#   阶段1 muvera 只吐 fde_topk 个候选页, 阶段2 maxsim 从其中取回 leg_topk。
#   若 fde_topk < leg_topk(去 config.ini 调的), 实际返回长度 = min(leg_topk, fde_topk),
#   Recall@leg_topk 会**静默**退化成 Recall@fde_topk —— 数字看着正常, 含义已经变了。
MM_K = int(conf.MM_LEG_TOPK)


def eval_multimodal(limit=None):
    """
    L4 单通道多模态检索: 文字 query -> ColQwen2 encode_text -> FDE+muvera
    粗排(阶段1) -> 本地精确 MaxSim 精排(阶段2) -> topK 页。纯本地, 免费。

    【为什么用 k=leg_topk 调一次就能算全档 Recall】image_leg(query, k) 内部就是
    阶段1 召回 fde_topk 候选页 -> 阶段2 对这批做 MaxSim 并取前 k。
    所以一次调用返回的完整排序, 就够算 Recall@1/5(leg_topk) 了 —— 不必分两次跑。
    """
    from rag_qa.core.image_leg import PageRetriever

    with open(EVAL_SET, encoding='utf-8') as f:
        rows = [r for r in json.load(f) if r['gold_pages']]
    if limit:
        rows = rows[:limit]
    print(f'\n多模态检索评估(单通道文字 query): {len(rows)} 条(有 gold 的题)')

    retriever = PageRetriever(load_on_init=True)

    # 预热: ColQwen2 首次进显存 + Milvus collection load
    print('预热 ColQwen2(冷加载, 约 20~40s)...')
    retriever.image_leg(rows[0]['question'], k=MM_K)

    results = []
    t0 = time.perf_counter()
    for i, row in enumerate(rows, start=1):
        hits = retriever.image_leg(row['question'], k=MM_K)
        # image_leg 返回 (source_id, page, score); 真值单书 -> 投影回 int 页号。
        # id 必须跟着记: 下面要按 id 落盘原始证据, 没有 id 就只是无法对应的数字。
        results.append({'id': row['id'], 'ranked': ranked_to_ints(hits),
                        'gold': row['gold_pages']})
        if i % 10 == 0:
            print(f'  {i}/{len(rows)}  已用 {time.perf_counter() - t0:.0f}s')

    metrics = report_recall(results, [1, 5], 'L4 单通道多模态')
    metrics['seconds'] = time.perf_counter() - t0
    print(f'  用时 {metrics["seconds"]:.0f}s')

    # 原始证据落盘(report_recall 一印就把 ranked/gold 丢了)。
    #   落的是**全部** results(含被 report_recall 按 GOLD_MAX_PAGES 排除的题):
    #   排除规则将来若改了, 旧文件仍能按新规则重算, 落一份筛过的就做不到了。
    #   ⚠️ 带 limit 时必须换名(记忆教训: hardcoded-output-paths-overwrite-silently)——
    #   "名字描述了内容"这条不变量永远成立。limit=None(真跑)时叫 multimodal_raw.json。
    suffix = f'_limit{limit}' if limit else ''
    raw_path = os.path.join(EVAL_DIR, f'multimodal_raw{suffix}.json')
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'原始证据已落盘: {os.path.abspath(raw_path)}  ({len(results)} 条)')
    return metrics


def eval_refuse(n_per_side=30):
    """
    L4 拒答闸门的混淆矩阵 + 归因。

    【覆盖到哪儿 —— 别按"闸门数"读这张表】2026-09-29 用户裁决去掉了分数阈值闸门,
      单通道检索现在只剩两道闸门:
      · 闸门1(no_candidate) **按构造不可达**: 60 题都非空、检索恒有召回, 这个分支一次都走不到。
      · 闸门2(模型自评证据不足) **走过**, 才是这张表真正在度的 —— 唯一的真拒答闸门就是
        模型自评。不再有"分数 < MM_REFUSE_SCORE 被阈值拦"这一档(原闸门2 已删)。

    【数据来源】car_intent_5000.json 的「通用知识」是天然域外题,「专业咨询」是域内题,
      各抽 n_per_side 条, 看该答的答没答、该拒的拒没拒。

    ⚠️ 出网: 过闸门1(有候选)就调 qwen-vl 生成, 真花钱 —— 所以这张表**不能为了省钱
      把 reject 判读成卡分数**, 模型自评是唯一的兜。
    """
    from rag_qa.core.image_leg import PageRetriever
    from rag_qa.core.llm_client import get_llm_client
    from rag_qa.core.multimodal_qa import MultimodalQAService

    in_domain, out_domain = [], []
    with open(INTENT_DATA, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get('label') == '专业咨询' and 'page' in item:
                in_domain.append(item['query'])
            elif item.get('label') == '通用知识':
                out_domain.append(item['query'])

    random.seed(20260916)
    picked = {
        'in': random.sample(in_domain, n_per_side),
        'out': random.sample(out_domain, n_per_side),
    }

    svc = MultimodalQAService(retriever=PageRetriever(load_on_init=True),
                              llm=get_llm_client(),
                              redis_client=None)

    rows = []
    for side, questions in picked.items():
        for q in questions:
            try:
                result = svc.answer(q)
            except Exception as e:
                print(f'  异常({side}): {q} -> {type(e).__name__}: {e}')
                continue
            # 判据必须是 **source**, 不能用 hit。QAResult.refuse() 返回的是 hit=**True**
            #   (base/qa_result.py: 「拒答是 L4 的正常产出, 编排器不该再往下丢」)。
            #   写成 `not result.hit` 会让 60 条**全部**算成"没拒" -> 整个矩阵反转
            #   (误拒率 0/30、误放率 30/30), 而且看着像个漂亮结果, 不会报错。
            #   同一个坑在 new_main.py 的引用页清理处已经写死过一次, 别踩第二次。
            refused = (result.source == 'refuse')
            rows.append({'side': side, 'question': q, 'refused': refused,
                         'reason': result.meta.get('reason', '') if refused else ''})
            print(f'  [{side}] 拒答={refused} reason={rows[-1]["reason"]}  {q}')

    # ---- 混淆矩阵 ----
    fn = [r for r in rows if r['side'] == 'in' and r['refused']]     # 该答却拒 -> 误拒
    fp = [r for r in rows if r['side'] == 'out' and not r['refused']]  # 该拒却答 -> 误放
    n_in = sum(1 for r in rows if r['side'] == 'in')
    n_out = sum(1 for r in rows if r['side'] == 'out')

    # 分母要把**请求数**也印出来(n_in/n_per_side): 上面的 except-continue 会静默丢掉
    #   抛异常的题, 只印 n_in 的话, 少了 5 条就变成"域内 25 条"这种看不出少了人的分母;
    #   全军覆没时更会印出 `0/0 = 0.0%`, 长得跟通过一样。偏差方向永远是"更好看", 所以必须显形。
    print(f'\n拒答评估(域内 {n_in}/{n_per_side} 条 / 域外 {n_out}/{n_per_side} 条)')
    print(f'  误拒率(域内被拒)  {len(fn)}/{n_in} = {len(fn) / n_in * 100 if n_in else 0:.1f}%')
    print(f'  误放率(域外没拒)  {len(fp)}/{n_out} = {len(fp) / n_out * 100 if n_out else 0:.1f}%')

    # ---- 闸门归因 ----
    print('\n拒答归因(是哪道闸门拦的):')
    for reason in sorted({r['reason'] for r in rows if r['refused']}):
        cnt = sum(1 for r in rows if r['refused'] and r['reason'] == reason)
        print(f'  {reason or "(未标注)":20s} {cnt}')

    return {'n_in': n_in, 'n_out': n_out,
            # 有效样本数 = 真正拿到结果的那些题(== n_in + n_out)。上面那个
            #   except-continue 会丢掉抛异常的题, n_in/n_out 各自都带分母口径,
            #   这里再给一个总数, 好让基线头部一眼看出"60 条里实际跑成几条"。
            'n_valid': n_in + n_out,
            'false_refuse': len(fn) / n_in if n_in else 0.0,
            'false_accept': len(fp) / n_out if n_out else 0.0,
            'reasons': {r: sum(1 for x in rows if x['refused'] and x['reason'] == r)
                        for r in {x['reason'] for x in rows if x['refused']}}}


def eval_l2(n=100, rewrite=False):
    """
    L2 题库(BM25 over jpkb)。

    【黑盒判对错】BM25Search.search() 只返回**最像那一条的答案**, 不告诉你是哪条问题。
    所以判对错的办法是: 拿问题 q 去查, 回来的答案若等于 q 自己的答案, 就算命中 ——
    不伸手掏内部属性。

    【基线 vs 真指标】
      基线: 拿**原问题**查。问题原文就在库里, BM25 必然高分 -> **虚高**,
            只能证明"库和检索器是通的、索引没错位", 属 smoke 不是指标。
      真指标: 拿**改写后**的问题查(rewrite=True), 真值还是那条自己。
    """
    from mysql_qa.cache.redis_client import RedisClient
    from mysql_qa.db.mysql_client import MysqlClient
    from mysql_qa.retrieval.bm25_search import BM25Search

    redis_client = RedisClient()
    mysql_client = MysqlClient()
    bm25 = BM25Search(redis_client, mysql_client)

    all_questions = [row[0] for row in mysql_client.fetch_questions()]
    random.seed(20260916)
    picked = random.sample(all_questions, min(n, len(all_questions)))
    print(f'\nL2 题库评估: 抽 {len(picked)} 条(库共 {len(all_questions)} 条), '
          f'模式={"改写" if rewrite else "原问题基线"}')

    questions = rewrite_questions(picked) if rewrite else picked

    # 落盘: 把"改写句 <-> 它的源原题"配对存下来。两个用处 ——
    #   ① 改写这一趟要花 100 次付费调用, 不存的话事后想复核命中/漏检只能**再买一遍**;
    #   ② 漏检归因(排序对但被阈值拒 vs 真排错)可以直接从盘上复算, 不靠人眼。
    if rewrite:
        path = os.path.join(EVAL_DIR, 'l2_rewrites.json')
        with open(path, 'w', encoding='utf-8') as f:
            # query    = 库里那条原问题(真值锚点, 也是 truth 答案的出处)
            # rewritten= 实际拿去查 BM25 的句子
            json.dump([{'query': src, 'rewritten': rw}
                       for src, rw in zip(picked, questions)],
                      f, ensure_ascii=False, indent=2)
        print(f'-> {path}')

    hits, ranks, missed = 0, [], []
    for i, q in enumerate(questions, start=1):
        truth = mysql_client.get_answer(picked[i - 1])
        got, _flag = bm25.search(q)
        ok = (got is not None and got == truth)
        hits += ok
        # 两个参数别接反: q 是**拿去查的句子**(基线是原句, 改写趟是改写句),
        #   picked[i-1] 是**要定位的那一行**(恒为原问题)。只传一个的话, 改写趟必然
        #   拿改写句去 original_questions 里找, 永远找不到 -> 返回 None ->
        #   整批 None 时 MRR 静默变 0.0; 只坏一部分时 MRR 反被**抬高**(见下面那道 all(ranks) 闸)。
        ranks.append(l2_true_rank(bm25, q, picked[i - 1]))
        if not ok:
            missed.append((q, picked[i - 1]))
            print(f'  ✗ {q!r}')
        if i % 20 == 0:
            print(f'  {i}/{len(questions)}  累计命中 {hits}')

    hit_rate = hits / len(questions) if questions else 0.0
    # 排名里一旦掺进 None, 下面那句 `if r` 会把它**悄悄丢掉** —— 而丢掉的后果是把 MRR
    #   **抬高**、不是压低: 活下来的名次照常参与平均, 少掉几条差的, 均值只会更好看。
    #   整批 None 时 mean([]) 给 0.0(那还算显眼), 真正危险的是**只坏一部分**:
    #   数字比平时还漂亮, 读起来就是"排序没问题"。所以在这儿闸住, 不许它溜到屏幕上。
    #   `all([])` 为真, 空样本(n=0 / 空库)不会误炸。
    #   基线趟也走这里, 且它在**付费的改写趟之前** —— 语料缓存与 MySQL 漂了会先炸在这儿,
    #   一轮付费调用都不会花出去。
    assert all(ranks), (f'{sum(1 for r in ranks if not r)}/{len(ranks)} 条真值行在 BM25 全序里没找到; '
                        f'None 会被 mean([... if r]) 丢掉并**抬高** MRR, 不许静默通过')
    mrr = mean([1.0 / r for r in ranks if r])
    print(f'\nL2 top-1 命中率  {hits}/{len(questions)} = {hit_rate * 100:.1f}%')
    # 标签必须说清**这个排名是在哪个问句下算的**: 基线趟是拿原句查原句(恒为 1.0),
    #   改写趟才是拿改写句查、看真值那条排第几 —— 后者才有信息量。
    print(f'L2 MRR(诊断)     {mrr:.3f}   <- 用**{"改写后的问句" if rewrite else "原问句(自匹配, 恒等 1.0)"}**'
          f'查、看真值那条在全序里排第几; 绕开生产路径, 不是生产行为')
    if not rewrite and questions:
        # 不变量: 基线是"拿原句查原句"的真自匹配, 真值那条必然排第 1, MRR **必须**恰好是 1.0。
        #   **它钉的是数据, 不是参数**: 钉住"语料缓存原样存着抽出来的这批问题"——
        #   真值行排不到第 1, 或 original_questions 不是拍平的 str(如元组, `==` 永假),
        #   这个数就掉下 1.0; 那是**语料缓存与 MySQL 漂了**, 不是"这次排序不好"。
        #   ⚠️ 它**没有**发现"query/gold 两参接反"的能力: 基线趟 q == picked[i-1],
        #   两个参数互换是**空操作**, 换了照样 1.000。那类错只能由上面那道 all(ranks) 闸
        #   配合改写趟去暴露。
        #   `and questions` 不能省: 空样本(库为空/n=0)时 mean([]) 返回 0.0, 断言会**自己**
        #   炸掉 —— 那是拿"没抽到题"去栽赃数据漂移。没抽到题就没有不变量可谈。
        assert mrr == 1.0, (f'基线 MRR 应恒为 1.0(自匹配), 实得 {mrr} —— '
                            f'真值行没排到第 1, 先查语料缓存是否与 MySQL 漂了')

    # ---- 漏检归因: 分清"排序对但被阈值拒"与"真排错"。纯本地复算, 不花钱, 不改 hit_rate。 ----
    if missed:
        print(f'\n漏检归因({len(missed)} 条, 本地复算; 判据见 l2_miss_reason docstring):')
        tally = {}
        for q, gold in missed:
            reason, score = l2_miss_reason(bm25, q, gold, mysql_client.get_answer(gold))
            tally[reason] = tally.get(reason, 0) + 1
            print(f'  {reason}  softmax {score:.4f}  {q}')
        print('  ---- 小计 ----')
        for reason in ('排序对但被阈值拒', '真排错', '条目对但库里没答案'):
            if tally.get(reason):
                print(f'  {reason}  {tally[reason]} 条')

    if not rewrite:
        print('⚠️ 这是**自匹配基线**, 必然虚高, 只能当 smoke 读, 不是指标。')
    return {'n': len(questions), 'n_valid': len(questions), 'hit_rate': hit_rate,
            'mrr': mrr, 'rewrite': rewrite}


def l2_true_rank(bm25, query, gold_question):
    """
    真值那行在 BM25 全序里排第几(1 基); 找不到返回 None。

    ⚠️ 这是**诊断**, 它绕开了生产路径(生产只取 argmax)。它回答的是
    "拿 query 去查时, 答案所在那行排第几名", 用来判断 L2 该不该从 top-1 放宽到 top-k。

    **两个参数必须分开, 别合并成一个**:
      query        —— 真正送进 BM25 的句子。基线趟是原问题, 改写趟是**改写后的**。
      gold_question—— 要在 `original_questions` 里定位的那一行, **恒为原问题**。
    合并成一个的后果: 改写趟会拿改写句去 `original_questions` 里找它自己 —— 库里没有
    这么一行, 永远找不到 -> 返回 None -> 上面的 `except Exception` 再把它咽掉 ->
    **MRR 静默变成 0.0**, 而命中率照样好看, 看不出是接错了参数。
    """
    from mysql_qa.utils.preprocess import preprocess_text
    try:
        scores = bm25.bm25.get_scores(preprocess_text(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        for rank, idx in enumerate(order, start=1):
            if bm25.original_questions[idx] == gold_question:
                return rank
    except Exception:
        return None
    return None


def l2_miss_reason(bm25, query, gold_question, truth):
    """
    一条漏检**为什么**漏 —— 本地复算, 分清两种性质完全不同的失败。

      排序对但被阈值拒  : 正确条目**就排在 top-1**, 只是 softmax 没过 search() 的阈值,
                          于是 search() 返回 None。排序是对的, 卡在阈值上。
      真排错            : top-1 是**别的条目**, 检索真的排错了。
      条目对但库里没答案: top-1 是正确条目, 但那条在 jpkb 里没有答案(truth 为 None)。

    【为什么不用把 0.85 抄进来比一遍】
      只要 top-1 就是正确条目, 这条漏检就一定属于"被阈值拒" —— 因为 softmax 一旦达标,
      search() 会取那一条的答案, 而那答案正是 truth, 这一条就必然记为**命中**。
      于是"它漏了"这件事本身就已经证明了它没过阈值。
      把阈值抄进这里只会多出一个**与生产行为脱钩**的副本: 谁去调阈值, 这张表就开始说谎,
      而且抄错一个小数就会静默把某一类漏检算到另一类去。

    :return: (原因, top-1 的 softmax)。softmax 走 bm25._softmax —— 与生产**同一个**函数,
             拿到的是生产真正拿去和阈值比的那个数, 不是另算的一套。
    """
    from mysql_qa.utils.preprocess import preprocess_text
    scores = bm25.bm25.get_scores(preprocess_text(query))
    softmax_scores = bm25._softmax(scores)
    best_idx = max(range(len(softmax_scores)), key=lambda i: softmax_scores[i])
    best_score = float(softmax_scores[best_idx])

    if bm25.original_questions[best_idx] == gold_question:
        return ('条目对但库里没答案' if truth is None else '排序对但被阈值拒'), best_score
    return '真排错', best_score


def rewrite_questions(questions):
    """用云端 qwen 把每题换个说法, 去掉"问题原文就在库里"这个便宜。"""
    from rag_qa.core.llm_client import ERROR_PREFIX, get_llm_client
    llm = get_llm_client()
    out = []
    for i, q in enumerate(questions, start=1):
        prompt = (f'把下面这句用户提问换个说法, 保持意思不变, 只输出改写后的句子, '
                  f'不要任何解释:\n{q}')
        try:
            new_q = llm.chat_text(prompt).strip().strip('"')
        except Exception as e:
            print(f'  改写失败({e}), 用原句兜底: {q}')
            new_q = q
        # 上面那个 except 对**调用失败**其实是死代码: chat_text 从不抛异常, 它失败时
        #   返回 '错误: ...' 开头的字符串(llm_client.py:245-250)。不拦这个前缀的话,
        #   那句报错会被当成"改写后的问句"送进 BM25, 静默把命中率拉低, 日志里还看不出来。
        #   真正兜底的是这一句。
        if not new_q or new_q.startswith(ERROR_PREFIX):
            print(f'  改写失败(返回错误前缀), 用原句兜底: {q}')
            new_q = q
        out.append(new_q)
        if i % 20 == 0:
            print(f'  改写 {i}/{len(questions)}')
    return out


def eval_intent():
    """
    L3 意图分类的评估数由**重训时产出并落盘**(见 train_classifier.save_intent_result),
    这里只读出来打印 —— 重训一次是几十分钟的事, 评估台不该每次跑都重训。

    两个口径都留着: 随机划分的**虚高**(同页泄漏), 按页划分的才是主指标。

    ⚠️ **这一段是"读旧数", 不是"本次测出来的数"**:
      数在 `intent_result.json` 里, 由**上一次重训**产出。本次跑只是把它读出来。
      可见的痕迹是 `item['time']` —— 它与基线头部的 `meta.time` **不是同一天**
      (实测: L3 的 time 是 2026-09-16 16:08:10, 而 meta.time 是 2026-09-17 12:28:10),
      但光看 `sections_run` 里的 `L3_intent` 会以为这一段是本次跑的。
      ⇒ ①打印时带上读数的 `time`; ②返回 dict 里挂一条显式来源标记 `source`,
        让基线里这一段自己说清"这不是本次测的"。数值一个字都不动。
    :return: {划分口径: 读数, ..., 'source': 来源说明, 'n_valid': {划分口径: n_val}}
    """
    path = os.path.join(EVAL_DIR, 'intent_result.json')
    if not os.path.isfile(path):
        print(f'\nL3 意图分类: 还没有结果文件({path})。')
        print('  先按本计划 Task 7 Step 4 重训一次, 它会自己落盘。')
        return {}

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    print('\nL3 意图分类  (读 intent_result.json, **不是本次跑出来的**)')
    out = {}
    for key, label in [('random', '随机划分'), ('by_page', '按页划分')]:
        item = data.get(key)
        if not item:
            print(f'  {label:10s} (未跑)')
            continue
        note = '  <- 虚高, 不作主指标' if key == 'random' else '  <- 主指标'
        # time 一并印出来: 它跟基线头部的 meta.time 差一天, 是"这段是旧数"唯一看得见的证据。
        print(f'  {label:10s} accuracy={item["accuracy"]:.4f}  '
              f'macro_f1={item["macro_f1"]:.4f}  n_val={item["n_val"]}  '
              f'time={item.get("time", "(无)")}{note}')
        out[key] = item
    # 来源标记: 让基线自己声明这一段是缓存读数(别改数值, 这是已接受的既有局限)。
    out['source'] = 'cached reading of intent_result.json; not measured by this run'
    # 有效样本数: 两个口径的 n_val 不一样(随机 1000 / 按页 998), 所以这里是 dict 不是标量。
    out['n_valid'] = {k: v['n_val'] for k, v in out.items()
                      if isinstance(v, dict) and 'n_val' in v}
    return out


def judge_llm():
    """
    RAGAS 的裁判 LLM: 云端 qwen。**不用 Ollama**(thinking 模式慢 17 倍)。

    【这里为什么不需要 temperature 适配器】
      ragas 的指标走**异步**路径 —— 指标 -> prompt.generate() ->
      `ragas/llms/base.py` 的 `agenerate_text(temperature=1e-8)` ->
      langchain 的 `agenerate_prompt`。`ChatOpenAI._agenerate is BaseChatModel._agenerate`
      为 **False**(`BaseChatOpenAI` 自己定义了 `_agenerate`, langchain_openai/chat_models/base.py:1685),
      所以**不会**回退到同步的 `_generate`: 只重写 `_generate` 的话, 那个 `pop` 一次都执行不到。
      读源码 + 查 MRO 双重确认、并以冒烟实测印证: **DashScope 接受调用期 temperature**,
      四指标全有值、无 NaN。
      反例(**Ollama** 通道)不可平移: `ChatOllama` 没重写 `_agenerate`, 会走同步回退, 那时
      重写 `_generate` 才有意义。当前用 DashScope, 所以**别把这个子类加回来。**
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatOpenAI(api_key=conf.DASHSCOPE_API_KEY,
                                          base_url=conf.DASHSCOPE_BASE_URL,
                                          model=conf.LLM_MODEL))


def local_embeddings():
    """RAGAS 的 embedding: 本地 BGE-M3(免费、中文好)。

    ⚠️ 必须钉死 CPU(`device` 与 `devices` 都给 'cpu'): BGE-M3 默认会偷偷跑上 GPU(fp32),
    ColQwen2 4bit 已占大半显存, 4GB 卡装不下, 会在检索分发处 OOM。`devices`(复数)才真正
    管目标设备, `device` 只写回装饰性属性 —— 两者都得设。
    """
    from milvus_model.hybrid import BGEM3EmbeddingFunction
    from ragas.embeddings import LangchainEmbeddingsWrapper

    fn = BGEM3EmbeddingFunction(
        model_name_or_path=os.path.join(_PROJECT_ROOT, 'rag_qa', 'models', 'bge-m3'),
        use_fp16=False, device='cpu', devices='cpu')

    class _BgeM3:
        def embed_documents(self, texts):
            return [list(v) for v in fn(list(texts))['dense']]

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    return LangchainEmbeddingsWrapper(_BgeM3())


def patch_ragas_output_parser():
    """
    修 ragas 0.2.6 的输出解析 bug。**必须在 `evaluate()` 之前调用一次。**

    【bug 是什么】裁判输出格式不合法时, ragas 会调"修复提示词"把输出修好; 但那个修复提示词的
      输出模型是 StringIO(只有个 .text 字段), 而 `RagasOutputParser.parse_output_string`
      修完之后**把 StringIO 原样当结果返回, 从不重新解析成目标模型**。上层按真模型去取
      `answers.statements` → `AttributeError: 'StringIO' object has no attribute 'statements'`。
      本环境源码已逐行核实: `ragas/prompt/pydantic_prompt.py:379-382` 是
      `class FixOutputFormat(PydanticPrompt[OutputStringAndPrompt, StringIO])`,同文件 `:418`
      `result = fixed_output_string`、`:421` 原样 `return result`。

    【为什么无条件打上, 而不是等它出事】它把一个**静默的错误**换成**诚实的错误**:
      不打补丁时, 格式不合法会顺着 StringIO 一路走到 AttributeError, 被 executor 吞成 NaN,
      混进"有效样本 398/400"里 —— 你只知道少了 2 条, 不知道少在哪。打上补丁后, 修复出来的文本
      会被真正重新解析: 要么正常出分, 要么明确抛 `RagasOutputParserException` 把话说清楚。

    【出处与验证】`_archive/rag_qa/rag_assessment/rag_as.py:71-114`(同类前作, 实测过)。
      该补丁做过受控验证: 用桩 LLM 并排跑, 原版返回 StringIO → AttributeError,
      补丁版返回正确的输出模型; 30 样本 / 120 job 的实测里失败数 3 → 1, 剩下那 1 个是
      诚实报错的 `RagasOutputParserException`。

    【跟 ragas 版本绑死】这是给 ragas 内部补丁, 升级 ragas 时这条要重新核。若某天
      `ragas.prompt.pydantic_prompt` 里已经没有 `parse_output_string` 这个名字(上游修了),
      这个函数会当场报 ImportError —— 那是好事, 说明可以删掉它了。
    """
    from langchain_core.exceptions import OutputParserException
    from langchain_core.output_parsers import PydanticOutputParser
    from ragas.callbacks import new_group
    from ragas.exceptions import RagasOutputParserException
    from ragas.prompt.pydantic_prompt import (
        RagasOutputParser, OutputStringAndPrompt, fix_output_format_prompt,
    )
    from ragas.prompt.utils import extract_json

    async def parse_output_string(self, output_string, prompt_value, llm,
                                  callbacks, retries_left=1):
        callbacks = callbacks or []
        try:
            return PydanticOutputParser.parse(self, extract_json(output_string))
        except OutputParserException:
            if retries_left == 0:
                raise RagasOutputParserException()
            retry_rm, retry_cb = new_group(
                name='fix_output_format',
                inputs={'output_string': output_string},
                callbacks=callbacks,
            )
            fixed = await fix_output_format_prompt.generate(
                llm=llm,
                data=OutputStringAndPrompt(
                    output_string=output_string,
                    prompt_value=prompt_value.to_string(),
                ),
                callbacks=retry_cb,
                retries_left=retries_left - 1,
            )
            retry_rm.on_chain_end({'fixed_output_string': fixed})
            # 关键差异: 原版是 `return fixed`(返回 StringIO);
            # 这里拿修复后的文本**重新解析**, 递归直到解析成功或重试次数用尽。
            return await parse_output_string(
                self, fixed.text, prompt_value, llm, callbacks, retries_left - 1)

    RagasOutputParser.parse_output_string = parse_output_string


def eval_generation_text(limit=None, strict=False):
    """
    纯文字路径的生成质量: RAGAS 四个指标。

    【strict 是干嘛的】ragas 默认把跑挂的样本吞成 NaN(raise_exceptions=False)。
      **全量跑必须用这个默认值** —— 一条坏样本不该毁掉整批 100 题的 API 花费。
      但代价是你只知道"有效 398/400", 不知道那 2 条**为什么**没了。
      所以先拿 3 条 strict=True 跑一遍: 第一条坏样本当场抛异常, 把真正的原因喊出来。

    【contexts 取的是"检索回来的页", 不是"答案引用的页"】
      context_precision / context_recall 问的是"检索回来的原文里有没有用/漏没漏", 所以 contexts
      必须是**送进模型的那批候选页**; 若用 cited_pages 就变成拿答案去证答案, 循环了。

    【候选页从 meta['candidates'] 拿, 不需要二次检索】它**就是**真正送进模型的那批页号
      (`multimodal_qa.py:254` pages_sent 只记页图真存在、真送出去的页 -> `:289` 塞进
      rag_answer 的 meta -> `qa_result.py:58` 把 meta 原样收进 result.meta)。早先那版在这里
      又调一次 retriever.retrieve 是**多余且更不准**的: 它不知页图缺失, 把"模型根本没看到的页"
      也算成候选; 删它还省了每题一次检索(约 11s)。

      ⚠️ **answer() 必须先调、contexts 后算**。顺序反过来不是静默降级: `result` 未赋值时
        `result.meta` **当场**抛 UnboundLocalError —— 很吵, 一眼看出顺序写反, 不会假装成
        "空上下文"。(本机最小复现实测, 出处 task-9-report.md。)

    【空上下文会静默记 0.0, 不是 NaN(本地复算实测 [] -> 0.0, 非 NaN)】两个后果:
      ① 空上下文不被 summarize_ragas 的 NaN 检查拦下, 会静默拉低均值 → 必须数出来暴露
         (n_empty_context); 而且本项目 354 页里有 20 页无文字层, 这是现实风险。
      ② 各"好"情形 4 位小数下全打印 `1.0000` ⇒ 这个指标**只反映排序、不反映有用页数量**。
    """
    from rag_qa.core.image_leg import PageRetriever
    from rag_qa.core.llm_client import get_llm_client
    from rag_qa.core.multimodal_qa import MultimodalQAService
    from ragas import evaluate, EvaluationDataset, RunConfig
    from ragas.metrics import (faithfulness, answer_relevancy,
                               context_precision, context_recall)

    with open(EVAL_SET, encoding='utf-8') as f:
        rows = [r for r in json.load(f) if r['gold_pages'] and r['reference']]
    if limit:
        rows = rows[:limit]
    print(f'\n生成评估(纯文字): {len(rows)} 条')

    pages = page_text_map()
    svc = MultimodalQAService(retriever=PageRetriever(load_on_init=True),
                              llm=get_llm_client(), redis_client=None)

    samples = []
    refused = 0
    n_empty_context = 0
    for i, row in enumerate(rows, start=1):
        result = svc.answer(row['question'])   # 先 answer —— 候选页要从它的 meta 里取
        if result.source == 'refuse':
            refused += 1                       # 拒答没有答案正文可评, 跳过
            continue
        # 候选页取 meta['candidates'] —— 它是真正送进模型的页号(出处见 docstring),
        # 比再检索一次更准: 页图缺失、模型没看到的页不会被算进来。
        candidates = result.meta.get('candidates', [])
        # 只留有文字层的页 —— 354 页里有 20 页没有文字层, 那些页给不出候选原文。
        # candidates 现在是 (source_id, page) 元组列表 -> 真值单书, 投影回 int 页号取文字层
        contexts = [c for c in (pages.get(q, '') for q in ranked_to_ints(candidates)) if c]
        if not contexts:
            # 空上下文 ragas 记 0.0 而不是 NaN(见 docstring 那张表), 那道 NaN 检查拦不住它。
            n_empty_context += 1
        samples.append({
            'user_input': row['question'],
            'response': result.answer,
            'retrieved_contexts': contexts,
            'reference': row['reference'],
        })
        if i % 10 == 0:
            print(f'  {i}/{len(rows)}  已收集 {len(samples)} 条'
                  f'(拒答 {refused} 条跳过, 空上下文 {n_empty_context} 条)')

    if not samples:
        print('❌ 一条样本都没收集到, 检查是不是全被拒答了')
        # 早退也要带计数(Task 8 的收尾, 控制器裁决 Ruling 78 授权):
        #   "跑了 98 条、98 条全被拒答" 跟 "这段根本没跑" 是两件完全不同的事, 后者才是 bug。
        #   返回 {} 的话 Task 11 的汇总里只剩一句"无数据", **分辨不出**是哪一种;
        #   带上计数就分得出, 而"全被拒答"本身是一个可行动的诊断。
        #   n_valid=0 显式写出来: 这一段**跑了但一个有效样本都没有**, 与"没跑"不同。
        return {'n_rows': len(rows), 'n_refused': refused, 'n_empty_context': n_empty_context,
                'n_valid': 0}

    print(f'收集完成: {len(samples)} 条进 RAGAS, {refused} 条因拒答跳过')
    if n_empty_context:
        # 让它显形: 这些行的 context_precision 会被 ragas 记成 0.0(不是 NaN),
        #   上面那道 NaN 检查抓不到它们, 混在均值里没人知道。
        print(f'  ⚠️ 其中 {n_empty_context} 条**一条有效上下文都没有**'
              f'(候选页全无文字层 / 没拿到候选页) —— 它们会把 context_precision 静默拉低')

    # 把样本落盘 —— Step 5 的人工抽检要拿它跟裁判分逐条对比; 不落盘就没法抽检。
    samples_path = os.path.join(EVAL_DIR, 'gen_text_samples.json')
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f'样本已落盘: {samples_path}')

    patch_ragas_output_parser()   # 必须赶在 evaluate() 之前, 见该函数的注释
    print('跑 RAGAS(慢, 每题十几次云端调用)...')
    # run_config 必须收紧。这不是"上个保险", 是算术:
    #   RunConfig 默认 max_retries=10 配 wait_random_exponential(max=60), 9 次重试的纯等待上限是
    #   1+2+4+8+16+32+60+60+60 = 243s; 而每个 (样本 × 指标) job 的超时默认只有 timeout=180s。
    #   光睡就把预算睡穿了。撞上超时的 job 会各占一个 worker 槽位直到超时(默认 16 worker),
    #   互相拖累, 最后交出一片 NaN。
    #   改成最多**尝试 3 次**(= 首次 + 2 次重试)、单次最多等 5s: 一个坏 job 很快认输让出槽位。
    #   措辞别写"重试 3 次" —— ragas 把 max_retries 喂给 tenacity 的 `stop_after_attempt`
    #   (`ragas/run_config.py` 的 add_retry/add_async_retry), 那是**尝试次数**不是重试次数;
    #   写成"重试 3 次"会让人以为总共能试 4 次。同理上面的默认值 max_retries=10 是 10 次尝试。
    #   (出处: _archive/rag_qa/rag_assessment/rag_as.py 文件头 [2], 该结论只跟 ragas 有关, 与 LLM 后端无关)
    result = evaluate(dataset=EvaluationDataset.from_list(samples),
                      metrics=[faithfulness, answer_relevancy,
                               context_precision, context_recall],
                      llm=judge_llm(), embeddings=local_embeddings(),
                      run_config=RunConfig(max_retries=3, max_wait=5),
                      raise_exceptions=strict)
    metrics = summarize_ragas(result, '纯文字路径')
    metrics['samples'] = samples_path
    # 三个计数并入返回 dict(原先只活在控制台): Task 11 聚合的是这个 dict, 只印不返回的话,
    #   "98 条里有几条被拒答/有几条上下文为空" 就落不进汇总。
    metrics['n_rows'] = len(rows)
    metrics['n_refused'] = refused
    metrics['n_empty_context'] = n_empty_context
    return metrics


def summarize_ragas(result, label, names=None, csv_name='gen_text_scores.csv'):
    """
    打印 RAGAS 结果, 并**强制检查 NaN**。

    【为什么 NaN 检查不能省】本项目踩过的最阴的坑: 四个指标全 NaN, 而 ragas
    默认把 NaN 静默丢掉 —— 看起来跑完了、报告也是空的, 不查还以为满分。

    【names / csv_name 为什么要参数化】(Task 9 修正案裁决 6)
      带图路径只算得动两个指标(faithfulness / answer_relevancy, 见 eval_generation_image),
      而它**落盘的文件名必须跟文本路径分开**: 两边都写 gen_text_scores.csv 的话, 后跑的
      那条会**静默覆盖**前一条 —— 那份 CSV 是人眼抽检材料的唯一数据源, 覆盖了就没有材料
      可摊开, 而且**不会报错**(同一个失效模式本轮真实发生过一次: 探针把
      `.smoke/task8_calls.json` 无条件写回同一路径, 清成了 `[]`)。
      默认值 = Task 8 的行为, 文本路径调它时一个字节都没变。
    """
    frame = result.to_pandas()
    if names is None:
        names = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    print(f'\n{label} —— RAGAS 结果')
    print(f'  总样本数 {len(frame)}')
    # n_valid = 进了 RAGAS 的样本数(== 总样本数)。**逐指标**的有效样本还会更少
    #   (ragas 会按指标各丢各的 NaN), 那些挂在 `<指标>_valid` 上, 见下面循环里那句注释。
    out = {'n': len(frame), 'n_valid': len(frame)}
    for name in names:
        if name not in frame.columns:
            print(f'  {name:22s} **列不存在**')
            continue
        valid = int(frame[name].notna().sum())
        value = float(frame[name].dropna().mean()) if valid else float('nan')
        out[name] = value
        # 有效样本数并入返回 dict(原先只活在 stdout 那一行): 基线头部(spec §6.2)要记它,
        #   只印不返回的话落盘的 JSON 里就没有, 事后判断不出"这个均值是几个样本算出来的"。
        out[f'{name}_valid'] = valid
        print(f'  {name:22s} {value:.4f}   有效样本 {valid}/{len(frame)}')
        if valid < len(frame):
            print(f'      ⚠️ 有 {len(frame) - valid} 条是 NaN, 被 ragas 丢掉了 —— 别当成满分')

    # 逐条分数也落盘 —— Step 5 的人工抽检要拿它跟人眼判断逐条对照。
    # to_pandas() 的行序与传进去的 samples 顺序一致, 所以可以按下标对齐。
    # csv_name 由调用方给: 文本路径落 gen_text_scores.csv, 带图路径落 gen_image_scores.csv,
    #   两份互不覆盖(见 docstring)。
    frame_path = os.path.join(EVAL_DIR, csv_name)
    frame.to_csv(frame_path, index=False, encoding='utf-8-sig')
    out['scores'] = frame_path
    return out


def purge_l1(qa, picked):
    """
    先**报告**再**清冷**自己抽中那批题的 L1 缓存。必须在计时之前跑。

    【为什么"只清不报"和"只报不清"都不够】
      ① 只清不报: 下一次污染你依然看不见。你不知道这次测量原本是不是被自己上一轮
         写脏的, 于是"命中率突降"永远归因不到"上轮缓存被复用"。
      ② 只报不清: 这次数字**不可用**。`cache_key()` 里**没有 session_id、没有时间戳**
         (`base/qa_result.py:178-192` 就是 `'qa:' + md5(normalize_query(query))`),
         所以 `session_id='eval-run'` **完全隔离不了缓存** —— 同一种子抽出的同一批题,
         上一轮 eval_e2e 写进去的答案这一轮会**全部命中**, 报告上就是
         "命中率近 100% + 延迟近 0", 而它测的是缓存不是系统。
      两条**都要做**: 先算出 pre_cached 并打印、并进返回 dict, 再删, 然后才允许计时。

    【删哪些 key】**只删本函数抽中的那几十条**, 且用 `cache_key()` 现算 ——
      与生产读写**同一个函数**。自己拼 `'qa:' + md5(...)` 的话, 归一化规则一旦跟
      `normalize_query` 差一点(全角标点/大小写), 就**一条都删不掉而且不报错**,
      于是又回到"测的是缓存"。**绝不清库**(flushall 之类会把别人的缓存一起端掉)。

    【删完为什么还要当场复核】"缓存没清掉"与"检索不准"在命中率上**长得一模一样** ——
      事后没法区分。删没删掉**只有这一刻**能无歧义地看到, 所以这一条断言不能省。

    :param qa: 编排器(app.qa_system)。只要有两个属性就能跑, 所以自检能拿桩替它。
    :return: pre_cached —— 抽中那批里**本来就在缓存里**的 id 列表
    """
    from base.qa_result import cache_key

    keys = {row['id']: cache_key(row['question']) for row in picked}
    pre_cached = [row['id'] for row in picked
                  if qa.redis_client.get_data(keys[row['id']])]
    print(f'\n清冷前(pre_cached): 抽中的 {len(picked)} 条里 {len(pre_cached)} 条本来就在 L1 缓存里'
          + (f'  -> {", ".join(pre_cached)}' if pre_cached else ''))

    for row in picked:
        qa.redis_client.client.delete(keys[row['id']])

    # 删完当场复核, 见 docstring 最后一段。`left` 非空 => 接下来的数字是缓存数不是系统数。
    left = [row['id'] for row in picked if qa.redis_client.get_data(keys[row['id']])]
    assert not left, (f'{len(left)}/{len(picked)} 条题的 L1 key 删完仍在缓存里: {left} —— '
                      f'接下来测的是缓存不是系统, 别把下面的数字当指标读')
    return pre_cached


def eval_e2e(n=30):
    """
    端到端: 走编排器 app.qa_system, 不起服务。

    ⚠️ 会往 conversations 写数据(正常提问的写入, 不是清库), session 是
    `eval-run-<本次运行标签>-<题号>`(每题一个), 仍以 eval-run- 开头。
    跑完**不删** —— 删了就没法在页面上复看。

    【session 为什么是 `eval-run-<id>` 而不是一个固定的 `eval-run`】
      `answer()` 每次都取该 session **最近 5 轮**历史(`new_main.py:297` 的
      `_fetch_recent_history(session_id, limit=5)`)。固定成一个 session 的话,
      n=30 就从第 2 条起带着前几轮的问答、第 6 条起带满 5 轮 —— 那就**不是"30 条独立查询",
      而是"一段 30 轮的对话"**: 行与行之间不再可交换, 30 条也就不是 iid 样本,
      而我们报的"引页 ∩ gold"命中率是这个样本上的估计。
      历史确实会改被引页: `cited_pages` 可以来自生成路径(见 `new_main.py:523` 那句
      "顺序固定: 先认生成路径, 再认缓存路径"), 生成是看着历史答的。
      代价还在别处现形: 多轮时 L4 **跳过缓存回填**(日志 `[L4] 多轮场景, 跳过缓存回填`),
      于是 L1 只被第一条写脏 —— 同一现象的另一面。
      ⇒ 每题各自成会话(`eval-run-0916-174500-q086`), 历史为空, 行与行才是可交换的。
      **仍带 `eval-run-` 前缀、仍写进 conversations、跑完仍不删** ——
      spec/plan 要的只是"能复看"这一条, 它们从未讨论过历史累积的后果。

    【为什么还要带「本次运行」标签】
      上一段解决的是**同一次运行内部**的行间污染; 这一段解决**跨运行**的:
      `answer()` 只按 session_id 取历史, session_id 里若不带本次运行的标识,
      第二次跑**同一个** session 就会**静默继承上一次的历史** —— 行不再独立,
      而报告上**一点痕迹都没有**。这一条同样不会被 `purge_l1` 挡住:
      它只清 **Redis**, 而历史在 MySQL 的 `conversations` 表里, 两者管的不是一回事
      (控制器实测: 库里已有 `eval-l1` 2 轮、旧 `eval-run` 3 轮的残留,
      见 task-10-fix2.md 第零节; `purge_l1` 一条都动不了它们)。
      ⇒ 标签用 `%m%d-%H%M%S`(**带秒**: 同一分钟内跑两次也不会撞), 且只能加在
      `eval-run-` **之后** —— 前缀不能动, spec 要的"能复看"认的就是它。
      ⇒ 一次运行 = 一组标签: 这批 30 条跑完, 在库里按标签就能整批捞出来复看。

    ⚠️ 端到端**不评拒答**: 拒答是 L4 的正常产出, 端到端只关心
    "该答的题有没有给出引用了正确页的答案"。

    【为什么用 `answer_with_meta` 而不是 `query_with_meta`】
      两者**等价**: `new_main.py:830` 的 `answer_with_meta` 就是 `return self.answer(...)`,
      而 `query_with_meta` 只是把同一个 result 按 24 字切片 yield 再收尾带上 cited_pages。
      端到端**本来就不评打字机**, 而 `answer_with_meta` 直接给出 `result.source` ——
      那是 `layers` 这一列的**唯一来源**。走 query_with_meta 的话只能从收尾块里掏
      `cited_pages`, **拿不到 source**, 于是"这 30 条是不是缓存答的"就无从判断。

    【为什么必须先 purge_l1 再计时】见 purge_l1 的 docstring: 不清冷就是拿缓存当系统测。

    【为什么必须报 `layers`】"命中率 97%" 与 "命中率 97%, 但其中 30 条是缓存答的"
      在报告里**长得一模一样**。缓存命中会同时表现为 ①`layers` 里冒出 `'redis'` 这一档,
      且 ②该行因 `cited_pages` 为空被判成 miss(L1 命中时 `new_main.py:711` 的
      `extra = {'cited_pages': cited} if cited else {}` 可能根本不挂这个键)。
      所以 `layers` 是本次测量**唯一**的分层证据: 它既进返回 dict, 也**必须自己打印一行**。
      `print_summary` 现已支持 dict/list(见那函数 docstring 的第 ② 条), 全量跑汇总表里
      `layers {...}` / `misses [...]` / `reasons {...}` 都**实际印出来了**
      (日志 `.smoke/fullrun_0917_110532.out`)。那行 `print` 仍**保留** —— 它的作用是
      "带时间戳独立落在日志里", 可脱离汇总表复核。

    【`hit_rate` 是一个**混合总体** —— 必须把按层的拆分同时摆出来】
      全量跑实测 `layers = {'rag': 26, 'mysql': 4}`, 6 条 miss 里 4 条 `source=mysql`
      且 `cited=[]`。走 mysql(L2 题库)腿答的题**按构造不可能**命中引用页:
      `new_main.py:516` 的 docstring 自己写着「L2/L3 本来就没有页码」——
      `cited_pages()` 直接返回 `[]`, 于是 `set([]) & set(gold)` 恒空。那不是"引错了页",
      而是"这一层压根没有页可引"。
      ⇒ 30 条读出来是 rag 24/26 = 92.3%、mysql 0/4 = 0%, 平均 80%。把 80% 当
        "系统引页准确率"会**低估 rag 那一层**(它其实是 92.3%), 也会让人以为
        mysql 那层"答错了页"。
      ⇒ 所以 `hit_rate` 的计算与数值一个字节都不动, 但在它旁边**同时印**按层的拆分。
        这一层和上面 redis 那层一样, 是本函数唯一能区分"答得不好"与"根本没参与"的证据。
    """
    import app                       # 导入即建 qa_system(与 .smoke/page_cites_e2e.py 同法)

    with open(EVAL_SET, encoding='utf-8') as f:
        rows = [r for r in json.load(f) if r['gold_pages']]
    random.seed(20260916)
    picked = random.sample(rows, min(n, len(rows)))
    # 本次运行的标签: **整个循环只算一次**并在每行复用 —— 一次运行 = 一组,
    #   所以它标的是"哪一次跑的", 不是"哪一条题"(理由见 docstring「为什么还要带本次运行标签」)。
    run_tag = time.strftime('%m%d-%H%M%S')
    print(f'\n端到端评估: {len(picked)} 条(session=eval-run-{run_tag}-<题号>, 每题一个会话)')
    if not picked:
        # 早退也带全部键: 否则 Task 11 的汇总里"这段没跑"与"跑了但没数据"分不出来。
        print('❌ 一条带 gold 的题都没抽到, 端到端跳过')
        return {'n': 0, 'n_valid': 0, 'hit_rate': 0.0, 'p50': 0.0, 'p95': 0.0,
                'layers': {}, 'layer_hits': {}, 'pre_cached': [], 'misses': []}

    qa = app.qa_system
    pre_cached = purge_l1(qa, picked)      # 先报告、再清冷 —— 之后才开始计时

    hits, latencies, misses, layers, layer_hits = 0, [], [], {}, {}
    for i, row in enumerate(picked, start=1):
        t0 = time.perf_counter()
        # 每题一个 session + 本次运行标签 —— 理由见 docstring 那两段。
        result = qa.answer_with_meta(row['question'],
                                     session_id=f'eval-run-{run_tag}-{row["id"]}')
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        # `or []` 不能省: cited_pages **只在非空时才挂**(new_main.py:711),
        #   取不到就是 None, set(None) 会当场 TypeError。
        #  cited 现在是 "source|page" 引用串 -> 真值单书, 投影回 int 再与 gold 比。
        cited = result.meta.get('cited_pages') or []
        cited_ints = ranked_to_ints(cited if isinstance(cited, (list, tuple)) else [cited])
        ok = bool(set(cited_ints) & set(row['gold_pages']))
        hits += ok
        layers[result.source] = layers.get(result.source, 0) + 1
        # 命中数也得**按层**各记一份: 只知道总数 24/30 分不出"rag 24/26 + mysql 0/4"
        #   与"rag 20/26 + mysql 4/4" —— 而这两种情况的结论完全相反(见 docstring)。
        layer_hits[result.source] = layer_hits.get(result.source, 0) + int(ok)
        if not ok:
            # source 一并记下: Task 11 要按"空引用 / 错页"分桶, 而"是不是缓存答的"
            #   只有 source 能回答(拿不到 source 时这两件事分不出来)。
            misses.append({'id': row['id'], 'question': row['question'],
                           'cited': cited, 'gold': row['gold_pages'],
                           'source': result.source})
        print(f'  {i}/{len(picked)}  命中={ok} source={result.source} '
              f'cited={cited} gold={row["gold_pages"]} {elapsed:.1f}s')

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    # 最近秩百分位: 第 ceil(0.95n) 个样本(1 基) -> 下标 ceil(0.95n)-1。
    #   原写法 `int(n*0.95)` 在 0.95n 恰好是整数时会**偏大一格** —— n=20 时 int(19.0)=19
    #   直接退化成 max(第 20 个), P95 变成"最大的那一个"。当前 n=30 固定, 两者同值
    #   (ceil(28.5)-1 = 28), 所以这是个**潜伏**问题: 谁把 n 改成 20 的倍数就会中招,
    #   而 P95 变大看着像"延迟变差了", 不会有人怀疑是取法。clamp 到 n-1 是防 ceil 越界。
    p95 = latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)]
    print(f'\n端到端引用页命中率  {hits}/{len(picked)} = {hits / len(picked) * 100:.1f}%')
    # 命中率**旁边**必须同时给出按层拆分: 上面那个数是两个总体的混合(见 docstring)。
    #   别在这里改 hit_rate 的算法, 只是把构成摆出来。
    print('  按层拆分:  ' + ' | '.join(
        f'{src} {layer_hits.get(src, 0)}/{total} = {layer_hits.get(src, 0) / total * 100:.1f}%'
        + ('  (结构性为 0 —— L2 题库答案本就没有页码, 不算"引错页")' if src == 'mysql' else '')
        for src, total in sorted(layers.items())))
    print(f'端到端延迟  P50 {p50:.1f}s   P95 {p95:.1f}s')
    # 这行**保留**: 它带时间戳独立落在日志里, 可脱离汇总表复核(汇总侧现已支持 dict)。
    print('端到端分层计数(layers)  '
          + '  '.join(f'{k}={v}' for k, v in sorted(layers.items())))
    if misses:
        print('\n没命中的:')
        for m in misses:
            print(f'  [{m["id"]}] source={m["source"]} {m["question"]}  '
                  f'cited={m["cited"]} gold={m["gold"]}')
    return {'n': len(picked), 'n_valid': len(picked), 'hit_rate': hits / len(picked),
            'p50': p50, 'p95': p95, 'layers': layers, 'layer_hits': layer_hits,
            'pre_cached': pre_cached, 'misses': misses}


def check_l1_functional():
    """
    L1 缓存的**功能验证**(不是指标)。

    【为什么只验功能不报指标】静态测试集上"每题问两遍", 第二遍命中率必然 100% ——
    这个数只反映跑了两遍, 不反映任何真实情况。L1 的真实命中率只有线上流量能测。

    【为什么连第一遍的 source 也打出来】第一遍若**本来就是缓存命中**(这条题以前跑过),
      那"第二遍还在缓存里"什么都没证明。两个 source 并排看才分得清:
        · 第二遍 `source='redis'` 且 `first.source` 不是 —— 缓存写+读通道通了;
        · 两遍都是 `'redis'` —— 只证明了读, 写端这次没被考到。
      `cited` 为空只能说明**缓存里存的 payload 不带页码**: `_backfill_l1` 回填的 dict
      只有 answer/origin(`new_main.py:776`), 只有 L4 自己回填的才带 cited_pages。

    【第一遍之前必须**先把这条题的 L1 key 删掉**】(实测发生过两次: 全量跑日志
      1226–1233 行**两遍都是** `L1 命中(原始来源 l4_rag)` ⇒ 两遍 `source='redis'`,
      而 1235 行那句「第二遍有页码返回 = 缓存通道是通的」是**无条件**打印的 ——
      写端根本没被考到, 报告上却像验过。)
      这条 key 会在: 问句 `'怎样加热座椅？'` 正是 eval_set 的 q000, 而
      `eval_generation_text` / `eval_refuse` 传 `redis_client=None`(不写缓存), 所以它是
      **本次跑之前**留下的; `eval_e2e` 的 `purge_l1` 只删它抽中的 30 条, 够不着它。
      ⇒ ①删(用**生产同一个** `cache_key`, 别自己拼 —— 归一化差一点就一条都删不掉而且
        不报错); ②万一删完仍是 `redis`(别处运行把缓存再加热), **另印一行降级声明**,
        不许再让那句结论无条件成立。
      ⇒ 这里**不**像 `purge_l1` 那样"删不干净就断言炸掉": 那条闸门管的是**整段端到端**
        数字可不可用**(60 条全污染), 而本段只是功能验证 —— 一条 pre-existing 的缓存
        不该把整轮跑拦停; 降级声明已经把话说清楚了。

    【session 为什么是 `eval-l1-<本次运行标签>` 而不是写死的 `eval-l1`】
      两次调用**必须共用同一个 session** —— 那正是这段要测的东西(第二遍是否命中缓存);
      标签只负责把**本次运行**与**以往各次运行**隔开, 不负责把这两次调用隔开。
      不加标签的话, 这个写死的 session 会把历次运行的历史攒在一起(控制器实测库里
      已有 2 轮), 于是全量跑里这两问变成第 3 轮 —— 多轮时 L4 **跳过缓存回填**
      (日志 `[L4] 多轮场景, 跳过缓存回填`), L1 只被写脏一条, 本函数就会
      **假报"缓存通道没走通"**。`purge_l1` 挡不住: 它清的是 Redis, 历史在
      `conversations` 里。
    """
    import app
    from base.qa_result import cache_key

    qa = app.qa_system
    question = '怎样加热座椅？'
    run_tag = time.strftime('%m%d-%H%M%S')     # 一次运行一个标签, **两次调用共用**它

    # 第一遍之前先清冷(理由见 docstring): key 用**生产同一个** cache_key 算。
    key = cache_key(question)
    was_cached = bool(qa.redis_client.get_data(key))
    print(f'\nL1 功能验证前: 清掉本条的 L1 key (清前是否已在缓存里: {was_cached})')
    qa.redis_client.client.delete(key)

    first = qa.answer_with_meta(question, session_id=f'eval-l1-{run_tag}')
    second = qa.answer_with_meta(question, session_id=f'eval-l1-{run_tag}')
    cited = second.meta.get('cited_pages') or []
    print(f'\nL1 功能验证: 同一问题问两遍  '
          f'first.source={first.source}  second.source={second.source}  cited={cited}')
    # 这句结论**只在第一遍真的走了冷路径时才成立** —— 否则退化成无条件断言(见 docstring)。
    if first.source == 'redis':
        print('  ⚠️ 第一遍就命中缓存 ⇒ **写端本次未被考到**, 这条只证明了读端')
    else:
        print(f'  第一遍走的是冷路径(source={first.source}), 第二遍 source={second.source}'
              f' ⇒ 缓存的写端与读端都通了')
    if second.source != 'redis':
        print(f'  ⚠️ 第二遍的 source 是 {second.source!r} 而不是 \'redis\' —— 缓存通道没走通')
    if not cited:
        print('  ⚠️ 第二遍没拿到页码 —— 要么没走缓存, 要么缓存里存的东西不完整')
    return {'first_source': first.source, 'second_source': second.source,
            'second_pass_cited': cited,
            # 有效样本数: 本段就是"一个问题问两遍"这一件事, 所以恒为 1。
            'n_valid': 1,
            # 第一遍是不是**冷**的 —— 这是上面那句结论成不成立的唯一判据, 必须跟着落盘,
            #   否则事后只能从 first_source 反推, 而 first_source=='redis' 时
            #   分不出"缓存本来就在"与"缓存是这一轮别的段写热的"。
            'first_pass_was_cold': first.source != 'redis',
            # 清冷**之前**它是否已在缓存里(与 purge_l1 的 pre_cached 同义)。
            'pre_cached': was_cached}


# ============================================================
# 汇总与基线落盘 —— 各评估段的产物在这里收口
# ============================================================
# 全量跑应含的段(顺序 = main() 里的执行顺序)。
#   【为什么要有这份名单】报告得回答"哪几段**没跑**", 而光看 metrics 的键分不出
#   "没测"与"测了但没值": 没测的键**根本不在** metrics 里; 没测出值的键**在**
#   (值为空 dict)。(Ruling 119.1) 生成侧两段在一条样本都没收到时是**带计数早退**
#   (`eval_generation_text:847` / `eval_generation_image:1002`), 全文件**只有** `eval_intent`
#   (`:619`)会 `return {}`。⇒ 空段闸门仍然必要, 只是它真实覆盖的对象是 `eval_intent`。
#   全量跑刻意**不含 L2**(见 main(): `if args.only == 'l2'`), 这份名单是唯一说得出这件事的地方。
_ALL_SECTIONS = ['L4_multimodal', 'L2_baseline', 'L2_rewrite',
                 'L4_refuse', 'L3_intent', 'gen_text',
                 'e2e', 'l1_functional']


def snapshot_git_state(cwd=None):
    """
    取一次 git 状态并**冻结**成 {'commit':…, 'dirty':…}。由 `main()` 在开跑之前调一次。

    【为什么必须"开跑前冻结", 不能每次落盘现取】
      `collect_meta` 是**每落一段就调一次**的。现取的话, 它记的是"落盘那一刻的 HEAD",
      而不是"**真正执行的那一版**"。实测发生过: 那次 `--only retrieval` 约 **13:01:43**
      启动, 而 `3a1e66e` 是 **13:15:44** 才提交的(跑着的时候才提交), 可它写出的
      `baseline_20260917_retrieval.json` 里却标着 `3a1e66e` ——
      数字挂在一个**它从未执行过**的版本上。
      任何**跨提交的长跑**(全量跑实测要一小时以上, 而一次会话里提交好几笔是常态)都会这样,
      方向恰是"看起来更可信": 出处声明说了它撑不起的话, 拿这个哈希去 diff 会困惑"明明改了, 数字没动"。

    【为什么还要记 dirty】
      那次长跑启动时工作区**正带着未提交的改动**, 于是"跑的那一版连哈希都叫不出来"。
      这个布尔值把这件事**显形**(dirty=True); 不记的话只能靠事后去比时间戳。

    :param cwd: 取哪个目录的 git 状态。默认项目根; selfcheck 传一个**临时仓库**来验"脏"的
                判据本身(绝不去碰本项目仓库 —— 那是别人的工作区)。
    :return: {'commit': 短哈希或 'unknown', 'dirty': True/False/None}
             `dirty=None` 表示**取不到**(没装 git / 不是仓库) —— 不用 False 顶替:
             "不知道"与"干净"是两件事, 混成一个布尔就是把未知说成没事。
    """
    import subprocess
    cwd = cwd or _PROJECT_ROOT
    # stderr 一律丢掉: 失败**已经**由下面那两个哨兵值如实报出来了('unknown' / None),
    #   再让 git 往 stderr 喷一行 `fatal: ...` 只会污染评估日志 —— 而评估日志**就是证据**,
    #   干净跑里冒出一行 fatal 会让读的人以为出了事。这不是"静默吞异常": 吞掉的只有那行噪声,
    #   异常本身变成了记录在案的哨兵值。
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         cwd=cwd, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = 'unknown'
    try:
        # `--porcelain` 空输出 = 工作区干净。
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'],
                                             cwd=cwd, text=True,
                                             stderr=subprocess.DEVNULL).strip())
    except Exception:
        dirty = None
    return {'commit': commit, 'dirty': dirty}


# `main()` 开跑前冻结的那一版(理由见 snapshot_git_state 的 docstring)。
#   None = 还没冻结(直接调 collect_meta 的场合, 例如自检/手工调试) -> 那时才现取。
_FROZEN_GIT = None


def collect_meta(metrics=None):
    """
    基线头部必须有的几样东西。

    【为什么非有不可】没有这几项, 下次改完代码你**没法判断**"分数变了"
    是改动导致的还是别的原因(换了数据集 / 换了裁判模型 / 阿里云换版本)。

    `sections_run` / `sections_not_run` 按 `_ALL_SECTIONS` 与 metrics 的键算,
    进来的是"哪几段跑了"而不是"哪几段有值" —— 一个段跑完却一条样本都没收到时,
    它的键**在** metrics 里(值为空 dict), 这里就算"跑了"。

    【`valid_samples` —— spec §6.2 点名要的「有效样本数」】
      meta 记入**逐指标**的有效样本数 `有效样本 {valid}/{len(frame)}`(由 summarize_ragas 报上来),
      不止前四样(时间/commit/指纹/裁判模型): 光看基线 JSON 看不出"这个均值是几个样本算的",
      而 RAGAS 会**按指标各丢各的 NaN**(§5.5 那个最阴的坑), 于是每个指标的 n 都可能不一样。
      ⇒ 由各段自己报上来(`n_valid` + `<指标>_valid`), **不在这里重算**:
        各段"有效"的定义本来就不一样(检索腿要排除 gold 超 5 页的题, 生成侧要排除拒答的题),
        在汇总侧重算只会造出第二份可能与段内不一致的口径。
    """
    # 优先用 `main()` 开跑前冻结的那一份(git_commit / git_dirty, 理由见
    #   snapshot_git_state 的 docstring); 没冻结(直接调用的场合, 例如自检)才现取 ——
    #   兜底必须有, 否则自检与手工调试都会撞 AttributeError。
    git_state = _FROZEN_GIT or snapshot_git_state()

    fingerprints = {}
    for name, path in [('eval_set', EVAL_SET)]:
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                fingerprints[name] = hashlib.md5(f.read()).hexdigest()[:12]

    ran = set(metrics or {})
    valid_samples = {}
    for section, value in (metrics or {}).items():
        if not isinstance(value, dict):
            continue
        picked = {k: v for k, v in value.items()
                  if k == 'n_valid' or k.endswith('_valid')}
        if picked:
            valid_samples[section] = picked
    return {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_commit': git_state['commit'],
        # 开跑那一刻工作区是否不干净: True = 跑的这版**没有哈希能指代**(改动还没提交)。
        'git_dirty': git_state['dirty'],
        'dataset_fingerprint': fingerprints,
        'judge_model': conf.LLM_MODEL,
        'embedding': 'local BGE-M3',
        'valid_samples': valid_samples,
        'sections_run': [s for s in _ALL_SECTIONS if s in ran],
        'sections_not_run': [s for s in _ALL_SECTIONS if s not in ran],
    }


def baseline_path(only=None):
    """
    本次要写的基线文件**绝对路径**(main() 与 save_baseline 共用这一个算法)。

    【为什么必须抽成一个函数】F6 的防呆闸在 `main()` 开头查"这个路径在不在",
    而真正写它的在 `save_baseline` —— 两处各写一遍命名规则的话, 谁改了一处,
    另一处就会去查/写**另一个文件**, 于是闸门形同虚设且不报错。

    【命名规则(裁决 2, 别改)】全量 `baseline_YYYYMMDD.json`, `--only X` 带后缀
    `baseline_YYYYMMDD_X.json`。只带日期的话, "同日先跑全量、再跑 `--only l2`"
    会用只含 L2 的文件**静默覆盖**全量基线。

    ⚠️ **已知边界**: 日期是**每次调用现取**的, 所以一次**跨午夜**的长跑(全量跑实测要几小时)
    会让 `main()` 开头查的路径与后来真正写的路径**不是同一个** —— 闸门查的是"今天"那份,
    最后落在"明天"那份上。后果是"换个日子就不受闸门保护", 而不是数据被毁。
    真要在午夜附近跑长任务, 先确认隔天那个文件名也不存在。
    """
    suffix = f'_{only}' if only else ''
    return os.path.join(EVAL_DIR, f'baseline_{time.strftime("%Y%m%d")}{suffix}.json')


def save_baseline(all_metrics, only=None):
    """
    基线落盘, 返回写进去的 payload(`{'meta':…, 'metrics':…}`)。
    **每跑完一段就调一次**(累计覆盖**同一个**文件), 末尾再调一次。

    【为什么不是"末尾调一次就够"】`try/except` 接不住**进程被杀死** —— 本项目实测发生过
    冷启动硬崩(EXIT=139 / 零输出, 见记忆 cold-start-torch-hard-crash), 而评估台这几个段
    都要加载 torch 模型 ⇒ 中途 segfault 是现实风险。一旦发生, 内存里的 metrics 全丢,
    而这些数是**逐条付费**换来的, 用户明说"只买一次"。
    ⇒ 每个段边界落一次, 文件就**永远**反映"到此为止跑成的全部内容"; 跑到一半去看它,
      拿到的是一份可用的**部分报告**。基线 JSON 是唯一不可替代的落盘物
      (汇总表不是 —— 崩了就没有表, 但报告可以从这份 JSON 重建)。

    【为什么返回 payload 而不是路径】(F8/M4) 把写进去的那份 `meta` 原样返回、交给
    `print_summary`, 让它**不再自己重算一遍** `collect_meta` —— 否则印在屏幕上的
    时间/`sections_*` 可能与文件里那份**不是同一份**(两次调用之间 `sections_not_run`
    会因 metrics 增长而变), 屏幕与文件就永远一致。路径本来就被本函数自己印出来了。

    【为什么写成"临时文件 + os.replace"】(F8/M1) 原来是 `open(path,'w') + json.dump`:
    一旦死在那个写窗口里(上面说的 segfault 是现实风险), 留下的是**半截 JSON** ——
    而这份文件是**唯一不可替代**的落盘物, 解析不了就等于把之前几小时付费买来的数全毁了。
    `os.replace` 在同一文件系统内是**原子**的(临时文件放同目录就是为了保证同盘):
    要么整个新文件可见, 要么整个旧文件还在, 不存在中间态。

    【F7: 段名单不许与 main() 脱钩】`_ALL_SECTIONS` 与 `main()` 是手工平行的两份名单,
    谁往 main() 加了段却忘了加进名单, 那段会在 `sections_run` 与 `sections_not_run`
    里**同时消失** —— 正是空段闸门要防的"报告看着完整、一整段没了"。在这儿断言住。
    """
    unknown = set(all_metrics) - set(_ALL_SECTIONS)
    assert not unknown, (
        f'这些段不在 _ALL_SECTIONS 名单里: {sorted(unknown)} —— '
        f'main() 里加了段却忘了往名单里加的话, 它会在 sections_run 与 sections_not_run '
        f'里**同时消失**, 报告看着完整而一整段没了。补救: 按 main() 里的执行顺序把段名'
        f'补进 _ALL_SECTIONS(`{__name__}` 靠它算哪几段跑了/没跑)。')

    path = baseline_path(only)
    payload = {'meta': collect_meta(all_metrics), 'metrics': all_metrics}
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        # 中途抛异常时把半截临时文件清掉(成功路径上它已经被 replace 走了, 这里不命中)。
        #   进程被 **kill** 时这行跑不到 —— 那种情况会留下一个 .tmp, 无害:
        #   真正的基线文件仍是完整的那一份旧文件。
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print(f'\n基线已落盘: {path}')
    return payload


def print_summary(all_metrics, meta):
    """
    汇总表。**只打印**, 落盘是 `save_baseline` 的事。

    `meta` 必须由调用方传**刚写进文件的那一份**(`save_baseline` 的返回值),
    **不许在这里再 `collect_meta()` 一次**: 那是两次独立的计算, 中间的秒级时间差与
    指标增长会让屏幕上的时间/`sections_not_run` 与文件里的 `meta` 对不上 ——
    "屏幕说有 3 段没跑、文件里写着全跑了"这种矛盾没有哪个检查会报出来。

    【两条静默丢弃路径必须堵上】
      ① 空段不许 `continue` 掉: `eval_intent` 在结果文件不在时**正是 `return {}`**
         (`:619`) —— 跳过它, 报告会**看起来完整**, 而一整段静默消失。⇒ 显式打成「无数据/失败」。
         (实测代码: 生成侧两段"一条样本都没收到"时**不是** `return {}`, 而是**带计数早退**
          (`{'n_rows':…, 'n_refused':…, 'n_empty_context':…}`, Ruling 78)。那三个计数会被正常打出来,
          恰好分得清"全被拒答"与"根本没跑"; 本分支守的是**别的**空段, 例如 `eval_intent`。)
      ② dict / list 值必须打得出来: 否则 `check_l1_functional` 的 `second_pass_cited`、
         `eval_e2e` 的 `layers`(端到端分层计数 —— 唯一看得出"测的是不是系统"的东西)
         与 `misses`、`eval_refuse` 的 `reasons` **全部**看不见。⇒ 用一行 JSON 打出来。
      (注: `eval_e2e` 里那句"汇总侧印不出 dict"在本函数支持 dict 后已过时, 但那行 print
       本身仍有价值 —— 它带时间戳独立落在日志里, 可脱离汇总表复核。)
    """
    print(f'\n{"=" * 72}\n评估汇总\n{"=" * 72}')
    for section, value in all_metrics.items():
        print(f'\n[{section}]')
        if not value:
            # 不是"未跑" —— 未跑是**根本不在** metrics 里, 见末尾那行
            print('  无数据/失败 —— 这一段跑了, 却一条样本都没收到')
            continue
        for key, val in value.items():
            if isinstance(val, float):
                print(f'  {key:24s} {val:.4f}')
            elif isinstance(val, (int, str)):
                print(f'  {key:24s} {val}')
            else:
                print(f'  {key:24s} {json.dumps(val, ensure_ascii=False)}')

    print(f'\n{"-" * 72}')
    print(f'  跑的时间   {meta["time"]}')
    # git_dirty 一并印出来: 它说的是"跑的那一版**有没有哈希能指代**"。
    #   不印的话, 这个字段就只活在 JSON 里, 屏幕上那份头部会显得比实际更可信。
    _dirty = meta.get('git_dirty')
    _dirty_note = ('  ⚠️ 开跑时工作区**不干净** —— 这一版没有哈希能指代' if _dirty
                   else ('  (取不到 git 状态)' if _dirty is None else ''))
    print(f'  git commit {meta["git_commit"]}{_dirty_note}')
    print(f'  数据集指纹 {meta["dataset_fingerprint"]}')
    print(f'  裁判模型   {meta["judge_model"]}')
    print(f'  embedding  {meta["embedding"]}')
    # 有效样本数(spec §6.2 点名要记的那一样)摆到屏幕上 —— 汇总视角只有这一句;
    #   其余只活在 summarize_ragas 的 "有效样本 X/Y" 那几行里。
    print(f'  有效样本数 {json.dumps(meta["valid_samples"], ensure_ascii=False)}')
    not_run = meta['sections_not_run']
    print(f'  本次未跑:  {", ".join(not_run) if not_run else "(无 —— 各段都跑了)"}')


# ============================================================
# 一、纯函数指标 —— 最容易写错、也最容易测的部分
# ============================================================
def recall_at_k(ranked_pages, gold_pages, k):
    """
    前 k 个里有没有 gold。命中任一 gold 即算对 —— gold 是"答案所在的页",
    不是"唯一正确页", 所以不做精确匹配。
    """
    return 1.0 if set(ranked_pages[:k]) & set(gold_pages) else 0.0


def reciprocal_rank(ranked_pages, gold_pages):
    """第一个 gold 页出现位置的倒数; 一个都没出现则为 0。"""
    gold = set(gold_pages)
    for i, page in enumerate(ranked_pages, start=1):
        if page in gold:
            return 1.0 / i
    return 0.0


def mean(values):
    return statistics.mean(values) if values else 0.0


def report_recall(rows, ks, label):
    """
    把一批 {ranked, gold} 汇总成一张 Recall@k / MRR 表。
    rows: [{'ranked': [页号...], 'gold': [页号...]}, ...]

    ⚠️ **`Recall@k` 的分母是"检索返回的那个池子", 而池子未必有 k 那么深。**
      `search(query, k)` 的 k 是**每路召回的子块数**, 管线 `dense+sparse -> 按页去重 -> reranker`
      返回**页级**(页号, 分); 同一页命中多子块, 去重后页数自然少于 k。实测 98 条文本腿池深分布
      `{8:2, 9:6, 10:12, 11:13, 12:23, 13:20, 14:16, 15:6}` —— 只有 6 条拿到满 15, 进指标的
      71 条里 **66 条(93.0%)池深 < 15**。机理另确认过: 加大 k、页数跟着涨 ⇒ 不是"只取前 k 页"
      (`(k,页)`=(15,11)/(30,25)/(60,47) 等三组)。所以池深不足的行, `Recall@15` 读作"返回池内
      有没有命中"。**这不是缺陷**: 按页去重是"按页评分"的必要前提, 是固有口径。`R@1/R@5` 不受
      影响(实测最小池深 8 ≥ 5), 但要**每次核对** → 下面都把池深摘要印出来并进返回 dict。
      ⇒ **别为"让 R@15 名副其实"去调 k 或改 search**: 那会改变已交付指标的语义, 需作为一次明确
         口径变更并重跑所有受影响的数(本轮只加标注, 一个数没动)。

    :return: 指标 + 有效样本数 + **池深摘要**(`depth_min` / `depth_median` / `depth_max` /
             `n_pool_lt_kmax`)。没有可用样本时池深三项为 None(不知道 ≠ 0)。
    """
    usable = [r for r in rows if len(r['gold']) <= GOLD_MAX_PAGES and r['gold']]
    dropped = len(rows) - len(usable)
    print(f'\n{label}  (参与 {len(usable)} 条, 排除 {dropped} 条: gold 为空或超 {GOLD_MAX_PAGES} 页)')
    # 池深摘要放在 Recall 行**之前**: 它限定的是下面整张表怎么读, 摆在后头就没人往回看了。
    depths = [len(r['ranked']) for r in usable]
    kmax = max(ks)
    if depths:
        depth_min, depth_median, depth_max = min(depths), statistics.median(depths), max(depths)
        n_pool_lt_kmax = sum(1 for d in depths if d < kmax)
        print(f'  池深(检索实际返回的页数) min {depth_min} / 中位 {depth_median:g} / max {depth_max}; '
              f'{n_pool_lt_kmax}/{len(usable)} 条的池深 < {kmax}')
        if n_pool_lt_kmax:
            print(f'    ⚠️ 那 {n_pool_lt_kmax} 条的 Recall@{kmax} 实际读作"**返回池内**有没有命中"'
                  f' —— 池子没那么深; k ≤ 最小池深({depth_min})的那几档不受影响')
    else:
        depth_min = depth_median = depth_max = None
        n_pool_lt_kmax = 0
        print('  池深: (没有可用样本)')
    for k in ks:
        vals = [recall_at_k(r['ranked'], r['gold'], k) for r in usable]
        print(f'  Recall@{k:<3d} {mean(vals) * 100:6.1f}%')
    mrr = mean([reciprocal_rank(r['ranked'], r['gold']) for r in usable])
    print(f'  MRR      {mrr:6.3f}')
    # n_valid = 有效样本数(进指标计算的那批)。检索腿上"有效"的定义就是"gold 非空且不超
    #   GOLD_MAX_PAGES", 所以它等于 n —— 但仍单独报一个名字: 汇总头部要的是**有名字的**
    #   有效样本数, 靠调用方从 n/dropped 里推的话, 换个口径就会悄悄变味。
    return {'n': len(usable), 'dropped': dropped, 'n_valid': len(usable),
            # 池深摘要也进 dict: 上面那行 print 会随日志滚走, 而"R@15 对多少行名不副实"这件事
            #   要能跟指标一起落进基线(理由与 n_valid 一样: 只印不返回 = 落盘里没有)。
            'depth_min': depth_min, 'depth_median': depth_median, 'depth_max': depth_max,
            'n_pool_lt_kmax': n_pool_lt_kmax,
            **{f'recall@{k}': mean([recall_at_k(r['ranked'], r['gold'], k) for r in usable])
               for k in ks},
            'mrr': mrr}


# ============================================================
# 二、自检 —— 纯函数先用小例子钉住, 再谈跑真链路
# ============================================================
def selfcheck():
    # ① gold 在第一位 -> Recall@1 命中, RR = 1.0
    assert recall_at_k([7, 3, 9], [7], 1) == 1.0, '① gold 排第一却没命中'
    assert reciprocal_rank([7, 3, 9], [7]) == 1.0, '① RR 应为 1.0'

    # ② gold 在第三位 -> Recall@1 = 0, Recall@3 = 1, RR = 1/3
    assert recall_at_k([7, 3, 9], [9], 1) == 0.0, '② Recall@1 应为 0'
    assert recall_at_k([7, 3, 9], [9], 3) == 1.0, '② Recall@3 应为 1'
    assert abs(reciprocal_rank([7, 3, 9], [9]) - 1 / 3) < 1e-9, '② RR 应为 1/3'

    # ③ 一个都没命中 -> 全 0
    assert recall_at_k([1, 2, 3], [9], 3) == 0.0, '③ 不该命中'
    assert reciprocal_rank([1, 2, 3], [9]) == 0.0, '③ RR 应为 0'

    # ④ 多 gold 页, 命中任一即算对 —— 这是本项目 gold 的语义, 别改成精确匹配
    assert recall_at_k([5, 6, 7], [6, 99], 2) == 1.0, '④ 命中任一 gold 即算对'

    # ⑤ RR 按**给定列表里的位置**算, 不做去重。两行是**同一形状**的两次: 一个**非 gold** 页
    #    在主命中之前重复一次, 把 gold 挤到第 3 位(第一行重复 3、gold 是 7; 第二行反过来)。
    #    "先去重再找位置"会把它算成 0.5, 按位置算是 1/3 —— 两行的期望值都必须是 1/3。
    #    写成 0.5 不是"更宽松", 而是在**钉住错误的去重实现**(0.5 正是去重版的答案)。
    #    注: "gold 页自身重复"那个形状**无区分力**(如 [3,7,7]/[7], 两种实现都返回 0.5), 故不用它;
    #    所以这里两行确实互为重复, 砍成一行也不损失覆盖。
    #    (订正史见账本 Ruling 16 / 17。)
    assert abs(reciprocal_rank([3, 3, 7], [7]) - 1 / 3) < 1e-9, '⑤ 重复页在前, gold 落在第 3 位应为 1/3'
    assert abs(reciprocal_rank([7, 7, 3], [3]) - 1 / 3) < 1e-9, '⑤ 同形状换一组页号, 仍应为 1/3'

    # ⑥ 空 gold 不该崩(会被 report_recall 排除, 但不能在算的时候炸)
    assert recall_at_k([1, 2], [], 2) == 0.0, '⑥ 空 gold 应为 0'

    # ⑦ summarize_ragas 的 NaN 分支 —— 用桩结果, 零出网零花费。
    #    为什么非要有这一条: 3 条冒烟里有效样本是 3/3, 这个分支**从没被真跑到过**,
    #    而"NaN 强制检查"是整个生成侧防线的地基(全 NaN 那次的教训)。不钉一次就没人验它。
    #    EVAL_DIR 要临时改到临时目录: 否则这一步会把真实那批 gen_text_scores.csv 覆盖掉。
    import io
    import shutil
    import subprocess                 # ⑫-⑥ 建临时 git 仓库验"脏"的判据
    import tempfile
    import types                      # ⑩ 造伪 app 模块用
    # 本模块对象: ⑫(换 _FROZEN_GIT)与 ⑭(换 time)都要**改自己的模块级全局**。
    #   必须在 ⑦ 就取好 —— 后面各条(⑫ 起)都要用, 放到 ⑬ 再取会 UnboundLocalError(踩过)。
    me = sys.modules[__name__]
    from contextlib import redirect_stdout
    from pandas import DataFrame

    global EVAL_DIR
    saved_dir = EVAL_DIR
    tmp_dir = tempfile.mkdtemp(prefix='eval_rag_selfcheck_')
    EVAL_DIR = tmp_dir

    class _StubResult:
        """冒充 ragas 的 EvaluationResult —— 只需要一个 to_pandas()。"""
        def __init__(self, frame):
            self._frame = frame

        def to_pandas(self):
            return self._frame

    try:
        stub = _StubResult(DataFrame({
            'faithfulness': [1.0, float('nan')],        # 2 条里 1 条 NaN
            'answer_relevancy': [0.5, 0.5],             # 全有效
            'context_precision': [float('nan')] * 2,    # **全 NaN** —— 最危险的那种
            'context_recall': [0.25, 0.75],
        }))
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = summarize_ragas(stub, '自检桩')
        printed = buf.getvalue()

        assert out['n'] == 2, f'⑦ n 应为 2, 实得 {out["n"]}'
        # 有 NaN 的列: 均值要**只用有效样本**算(1.0), 而且必须把 ⚠️ 打出来
        assert out['faithfulness'] == 1.0, f'⑦ 有 1 条 NaN 时均值应只用有效样本 = 1.0, 实得 {out["faithfulness"]}'
        assert '⚠️' in printed, '⑦ 有 NaN 却没打出 ⚠️ 警告 —— 这正是"看起来满分"那类静默错误'
        # 全 NaN 的列: 必须给出 nan, 不能给 0.0(给 0.0 就会被当成"这指标很差", 实际是"没测出来")
        assert out['context_precision'] != out['context_precision'], '⑦ 全 NaN 的列应返回 nan'
        # 没有 NaN 的列不该被无端警告
        assert out['answer_relevancy'] == 0.5 and out['context_recall'] == 0.5, \
            f'⑦ 无 NaN 的列算错了: {out["answer_relevancy"]}, {out["context_recall"]}'
        # **逐指标的有效样本数**必须落进返回 dict, 而且必须真是"该指标非 NaN 的条数"。
        #   为什么非要单独钉(复审 M1 实测出来的洞): `*_valid` 正是 I1(基线头部缺有效样本数)
        #   要修的那个口径, 而把它写成 `= len(frame)` 时**照样有值、照样是 int、
        #   屏幕上一模一样** —— 上一条 NaN 检查也照样过(它查的是均值, 不是这个计数)。
        #   ⇒ 唯一能分辨"该指标非 NaN 的条数"与"总样本数"的, 就是这个不等式:
        #     有 NaN 的列, 它的 `_valid` 必须**小于**帧长。只断等值(=1)的话,
        #     改回帧长(=2)也全绿 —— 那条断言没有区分力(复审已实测确认)。
        #   注意**别另立一条 `assert ... != 2`**: 它被上面那句 `== 1` 逻辑蕴含, 是**不可达断言**
        #   —— 永远不会响, 而它那段"为什么不能等于帧长"的话术也就永远印不出来。不可达的断言
        #   等于没有断言, 还留下"我守住了"的错觉(复审 R2-4 实测揪出来的)。所以那句话术**并进**
        #   这一条的 message: 真的要响的时候, 读的人一眼就看到"我记的是帧长而不是非 NaN 条数"。
        assert out['faithfulness_valid'] == 1, (
            f'⑦ faithfulness 有 1 条 NaN, 有效样本数应为 1, 实得 {out["faithfulness_valid"]}'
            ' —— 若它是帧长(2), 说明这个字段记的是"总样本数"而不是"该指标非 NaN 的条数", '
            '基线头部那一项就退化成一句恒等于 n 的废话; 而这个错不会让别的断言变红')
        assert out['answer_relevancy_valid'] == 2 and out['context_recall_valid'] == 2, \
            f'⑦ 无 NaN 的列有效样本数应为 2: ' \
            f'{out["answer_relevancy_valid"]}, {out["context_recall_valid"]}'
        assert out['context_precision_valid'] == 0, \
            f'⑦ 全 NaN 的列有效样本数应为 0, 实得 {out["context_precision_valid"]}'
        assert out['n_valid'] == 2, f'⑦ 帧级 n_valid 应为 2, 实得 {out["n_valid"]}'
        # 列不存在时不能崩, 也不能编一个值出来
        out2 = None
        with redirect_stdout(io.StringIO()):
            out2 = summarize_ragas(_StubResult(DataFrame({'faithfulness': [1.0]})), '自检桩-缺列')
        assert out2['n'] == 1 and out2['n_valid'] == 1 and out2['faithfulness_valid'] == 1, \
            f'⑦ 缺列时 n/n_valid/faithfulness_valid 都应是 1, 实得 ' \
            f'{out2.get("n")}/{out2.get("n_valid")}/{out2.get("faithfulness_valid")}'
        assert 'answer_relevancy' not in out2, \
            f'⑦ 缺列时应只在 out 里放存在的列, 实得 {sorted(out2)}'
        #   摘要行必须**跟上覆盖范围**。理由(复审 R2-1): 本轮交付物就是"覆盖补上了", 而下一轮
        #   复审唯一会读的证据**就是这一行 stdout** —— 只读自检输出、不看代码的人, 会因为
        #   "这里没提 *_valid" 而判 M1 没补、要求返工。⑫⑬ 那两行就是这么写的, ⑦⑩ 漏了。
        print('⑦ summarize_ragas 的 NaN 分支: 有 NaN / 全 NaN / 缺列 三种情形都对; '
              '另钉住逐指标有效样本数(*_valid 必须是非 NaN 条数, 不是帧长)与帧级 n_valid')
    finally:
        EVAL_DIR = saved_dir
        # 临时目录也要删掉。只还原全局变量是不够的 —— 那样每跑一次 --selfcheck 就往系统
        #   temp 里漏一个空壳目录(实测积了 3 个 eval_rag_selfcheck_*, 每个里一份
        #   gen_text_scores.csv 残留)。ignore_errors=True: 清理失败不该把自检本身弄挂。
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 钉住上面那句 rmtree: 清不掉就当场喊出来, 而不是等下一个人再去 ls 系统 temp 才发现。
    assert not os.path.exists(tmp_dir), f'⑦ 临时目录没被清掉, 又漏了一个: {tmp_dir}'

    # ⑧ summarize_ragas 的**参数化路口**(names / csv_name)。
    #    ⑦ 走的是默认参数, 覆盖不到这条路; 而这条路写错的后果是**静默覆盖**文本腿的
    #    gen_text_scores.csv —— 那份 CSV 是人眼抽检材料的唯一数据源(见该函数 docstring)。
    #    为什么钉在这儿而不是只靠 .smoke/ 那份离线脚本: scratch 不进版本控制, 被清掉就没有
    #    耐久回归了, 而这个失效模式恰恰是"不会报错"的那一类。零出网零花费(纯桩 + 临时目录)。
    tmp2 = tempfile.mkdtemp(prefix='eval_rag_selfcheck_')
    EVAL_DIR = tmp2
    try:
        with redirect_stdout(io.StringIO()):
            out3 = summarize_ragas(_StubResult(DataFrame({
                'faithfulness': [0.25],
                'answer_relevancy': [0.75],
                'context_precision': [0.5],      # 不在 names 里, 不该被报
                'context_recall': [0.5],
            })), '自检桩-自定义参数',
                names=['faithfulness', 'answer_relevancy'], csv_name='gen_zzz_scores.csv')

        # 先断"有没有覆盖"再断"有没有落对文件": 覆盖那条是真正的风险(静默、且毁掉人眼验收
        #   材料的唯一数据源), 让它在两条都成立时先喊出来, 报的病因更准。
        assert not os.path.exists(os.path.join(tmp2, 'gen_text_scores.csv')), \
            '⑧ 传了自定义 csv_name 却还是写了 gen_text_scores.csv —— 这会静默覆盖文本腿的产物'
        assert os.path.exists(os.path.join(tmp2, 'gen_zzz_scores.csv')), \
            '⑧ 自定义 csv_name 没被用来落盘'
        assert 'context_precision' not in out3 and 'context_recall' not in out3, \
            f'⑧ names 之外的指标不该出现在返回 dict 里: {sorted(out3)}'
        assert out3['faithfulness'] == 0.25 and out3['answer_relevancy'] == 0.75, \
            f'⑧ 自定义 names 的分数算错了: {sorted(out3)}'
        print('⑧ summarize_ragas 的参数化路口: 自定义 csv_name 落对文件, 且不碰 gen_text_scores.csv')
    finally:
        EVAL_DIR = saved_dir
        shutil.rmtree(tmp2, ignore_errors=True)

    assert not os.path.exists(tmp2), f'⑧ 临时目录没被清掉: {tmp2}'

    # ⑨ purge_l1 的两条性质: **先报后删**(pre_cached 必须看得见)与**删不干净就当场炸**。
    #    为什么非钉不可: "缓存没清掉"与"检索不准"在命中率上**长得一模一样**, 事后分不出来;
    #    而这条断言是唯一的防线 —— 它不响, 整段端到端数字就不可用。
    #    零出网零花费: 桩 qa, **不碰真 Redis**(真库是别人的生产缓存, 自检绝不许动它)。
    class _StubRedis:
        """只实现 purge_l1 用到的那两个入口: get_data / client.delete。"""

        def __init__(self, keys, delete_works=True, other=None):
            self.store = {k: {'answer': '桩答案', 'origin': 'l4_rag'} for k in keys}
            if other is not None:
                self.store[other] = {'answer': '别人的缓存', 'origin': 'l4_rag'}
            self.delete_works = delete_works
            outer = self

            class _Raw:
                def delete(self, key):
                    if outer.delete_works:
                        outer.store.pop(key, None)

            self.client = _Raw()

        def get_data(self, key):
            return self.store.get(key)

    class _StubQA:
        def __init__(self, redis_client):
            self.redis_client = redis_client

    OTHER_KEY = 'qa:别人的缓存_绝不能被删掉'
    # 桩里的 key 必须用**生产同一个** cache_key 算 —— 自己拼一个假 key 的话, 这个自检
    #   就变成了在验"我拼的 key 与我自己拼的 key 相等", 与生产完全脱钩。
    from base.qa_result import cache_key as _cache_key
    stub_picked = [{'id': 's001', 'question': '怎样加热座椅？'},
                   {'id': 's002', 'question': '胎压多少合适？'}]
    stub_keys = {r['id']: _cache_key(r['question']) for r in stub_picked}

    # 绿: s001 本来就在缓存里 -> 必须报出来、必须被删掉, 且**够不着别人的 key**
    stub = _StubRedis([stub_keys['s001']], other=OTHER_KEY)
    buf9 = io.StringIO()
    with redirect_stdout(buf9):
        got_pre = purge_l1(_StubQA(stub), stub_picked)
    assert got_pre == ['s001'], f'⑨ pre_cached 应报出本来就在缓存里的 s001, 实得 {got_pre}'
    assert 's001' in buf9.getvalue(), \
        '⑨ pre_cached 只进了返回 dict 没被打印 —— 下一次污染就永远看不见'
    assert list(stub.store) == [OTHER_KEY], \
        f'⑨ 清冷波及了不属于自己的 key(实测剩下 {sorted(stub.store)}) —— 只许删自己抽中的那几条'
    assert stub_keys['s002'] not in got_pre, '⑨ 本来不在缓存里的 s002 不该进 pre_cached'

    # 红(在本自检内部被接住): 删不掉时必须**当场炸**, 不许静默继续去测缓存
    stub2 = _StubRedis([stub_keys['s001']], delete_works=False)
    try:
        with redirect_stdout(io.StringIO()):
            purge_l1(_StubQA(stub2), stub_picked)
    except AssertionError as e:
        assert 'L1 key' in str(e), f'⑨ 炸了, 但炸出来的不是那道闸门: {e}'
    else:
        raise AssertionError('⑨ 缓存没清掉却没炸 —— 接下来测的会是缓存, 不是系统')
    print('⑨ purge_l1: 先报 pre_cached 再删、只删自己那批、删不干净当场炸')

    # ⑩ eval_e2e **每题一个 session** —— 零出网零花费(伪 app + 桩 qa, n=3, 不碰真 Redis)。
    #    为什么非钉不可: `answer()` 每次都取该 session 最近 5 轮历史
    #    (`new_main.py:297 _fetch_recent_history(session_id, limit=5)`)。固定成一个 session,
    #    n=30 就从第 2 条起带着前几轮的问答、第 6 条起带满 5 轮 —— **不是"30 条独立查询",
    #    而是"一段 30 轮的对话"**: 行不再可交换, 命中率也就不是 iid 样本上的估计。
    #    历史确实会改被引页(`cited_pages` 可来自生成路径, `new_main.py:523`)。
    #    spec/plan 写"固定 eval-run"时给的理由**只有**"写进 conversations 且跑完不删",
    #    从未讨论历史累积 ⇒ 那是没考虑副作用, 不是有意的多轮设计。
    #    这条断言必须能分辨"每题一个"与"全都一样": set 的长度就是判据。
    #
    #    ⚠️ 喂**假数据集**, 不喂真集: `eval_e2e` 是从盘上读 EVAL_SET 的, 直接跑就偷偷
    #    依赖"真集里带 gold_pages 的题 ≥ 3 条" —— 数据集缩水时它报的是
    #    "桩收到 2 次调用, 应为 3 次", **把数据问题报成了桩的问题**(评审 F1 实测)。
    #    自检撒谎比被测代码出错更危险: 你会照着它去改错地方。
    #    ⇒ 照 ⑦⑧ 的套路建临时目录写一份只含 3 条的假 eval_set.json(字段与真集一致),
    #      把模块级 EVAL_SET 指过去, 跑完**在 finally 里恢复**并把临时目录清干净。
    #
    #    还加一条与数据无关的: 三次调用必须**共用一个本次运行标签**、且**各自带本行 id**。
    #    理由: 标签将来若被改成"每题一个不同标签", 跨运行污染确实没了, 但
    #    "一次运行 = 一组"这个性质也没了 —— 而上面那条 set 长度断言**照样全绿**,
    #    没有任何东西会告诉你。判据只能是桩收到的 session 本身(零成本零出网)。
    class _StubE2EResult:
        def __init__(self, source, cited):
            self.source = source
            self.meta = {'cited_pages': cited}

    class _StubE2EQA:
        """只实现 eval_e2e 用到的两个入口: redis_client 与 answer_with_meta。"""

        def __init__(self):
            self.redis_client = _StubRedis([])   # 空缓存: pre_cached=[] 且删完无残留
            self.sessions = []

        def answer_with_meta(self, question, session_id=None):
            self.sessions.append(session_id)
            return _StubE2EResult('rag', [])

    fake_app = types.ModuleType('app')           # `import app` 命中它, 不建真编排器(也就零花费)
    fake_app.qa_system = _StubE2EQA()

    # 假数据集: 只 3 条带 gold_pages, 字段与真集一致(id/question/gold_pages)。
    #   id 里**不带 `-`**: 下面要把 session 切成"标签 + 本行 id"两段, 带 `-` 的 id
    #   会让切点在 id 内部(真集 id 形如 q086, 无此问题)。
    stub_rows = [{'id': 'q001', 'question': '怎样加热座椅？', 'gold_pages': [12]},
                 {'id': 'q002', 'question': '胎压多少合适？', 'gold_pages': [7, 8]},
                 {'id': 'q003', 'question': '空调不制冷怎么办？', 'gold_pages': [31]}]
    global EVAL_SET
    saved_set = EVAL_SET
    tmp3 = tempfile.mkdtemp(prefix='eval_rag_selfcheck_')
    with open(os.path.join(tmp3, 'eval_set.json'), 'w', encoding='utf-8') as f:
        json.dump(stub_rows, f, ensure_ascii=False)
    EVAL_SET = os.path.join(tmp3, 'eval_set.json')

    saved_app = sys.modules.get('app')
    sys.modules['app'] = fake_app
    try:
        # n=3 而不是默认的 30: 这条只验 session, 与样本量无关, 越小越快。
        with redirect_stdout(io.StringIO()):
            out10 = eval_e2e(n=3)
    finally:
        # 伪 app 必须撤掉: 留在 sys.modules 里会让**同一进程后续**的真 import app 拿到桩。
        if saved_app is None:
            del sys.modules['app']
        else:
            sys.modules['app'] = saved_app
        # EVAL_SET 也必须还原: 留在假数据集上, **同一进程后续**的真跑会读到只含 3 条的假集。
        EVAL_SET = saved_set
        shutil.rmtree(tmp3, ignore_errors=True)

    assert not os.path.exists(tmp3), f'⑩ 临时目录没被清掉: {tmp3}'
    assert EVAL_SET == saved_set, f'⑩ EVAL_SET 没还原: {EVAL_SET}'

    sessions = fake_app.qa_system.sessions
    assert len(sessions) == 3, f'⑩ 桩收到 {len(sessions)} 次调用, 应为 3 次(假数据集固定 3 条)'
    assert len(set(sessions)) == len(sessions), \
        f'⑩ 有题共用了同一个 session_id({sessions}) —— 历史会跨题累积, ' \
        f'"30 条独立查询"就变成了"一段 30 轮的对话"'

    # 拆开 session 看两件事: 标签**共用**、题号**各是各的**。
    #   形状: `eval-run-<本次运行标签 %m%d-%H%M%S>-<本行 id>`。
    #   为什么必须两条都断: 只断"共用"的话, 把题号写死成常量也照样绿; 只断"各不相同"
    #   的话(上面那条 set 长度), 每题一个不同标签也绿 —— 而那次运行就不再是一组了。
    shape = [re.match(r'^eval-run-(\d{4}-\d{6})-(.+)$', s or '') for s in sessions]
    assert all(shape), \
        f'⑩ session 形状应为 eval-run-<本次运行标签 %m%d-%H%M%S>-<本行 id> ' \
        f'(spec 只要求能写进 conversations 且不删, 但前缀必须仍是 eval-run-): {sessions}'
    tags = {m.group(1) for m in shape}
    assert len(tags) == 1, \
        f'⑩ 三次调用没共用一个本次运行标签(实得 {sorted(tags)}) —— ' \
        f'标签标的是"哪一次跑的", 一次运行必须是一组; 按题打标签虽然也隔开了跨运行历史, ' \
        f'但同一批题就散成了互不相干的组, 复看时捞不回来'
    assert {m.group(2) for m in shape} == {r['id'] for r in stub_rows}, \
        f'⑩ session 里的题号应各是本行 id({sorted(r["id"] for r in stub_rows)}), ' \
        f'实得 {sorted(m.group(2) for m in shape)}'
    assert out10['layers'] == {'rag': 3}, f'⑩ 分层计数没记对: {out10["layers"]}'
    # e2e 的**有效样本数**也要钉(复审 M1): 它是基线头部 valid_samples['e2e'] 的唯一来源,
    #   写死成 0(或换成任何与 len(picked) 无关的东西)不会让任何断言变红。
    #   判据必须是**具体的数字 3**(假数据集固定 3 条), 不能写成 `len(stub_rows)`:
    #   那样两边同源、一起变, 等于没测(复审点名了这一点)。
    assert out10['n_valid'] == 3, \
        f'⑩ e2e 的 n_valid 应为 3(假数据集 3 条全抽中), 实得 {out10["n_valid"]}'
    # 摘要行跟上覆盖范围(理由同 ⑦ 那条注释): 它还守着 `out10['n_valid'] == 3`, 不写出来
    #   的话, 下一轮只读自检输出的人会以为 e2e 的 n_valid 没人管。
    print(f'⑩ eval_e2e: 每题一个 session(eval-run-{sorted(tags)[0]}-<题号>), '
          f'三次调用共用一个本次运行标签、题号各是本行 id; '
          f'另一条: 返回的 n_valid == 3(桩里实际抽中的题数)')

    # ⑪ check_l1_functional 的两条性质(F2): ①清冷后第一遍**不是** redis;
    #    ②清不掉时(替身造出"第一遍就命中")必须**打出降级声明**, 不许再让那句结论无条件成立。
    #    为什么非钉不可: 实测在整轮全量跑里**两次都是** `first.source='redis'`
    #    (那条 key 是本次跑之前留下的), 而结论句照旧无条件打印 ——
    #    "写端没被考到"被读成了"缓存通道是通的"。报告上一点异常痕迹都没有。
    #    零出网零花费: 桩 app + 桩 redis, **不碰真 Redis**(真库是别人的生产缓存)。
    class _StubL1Result:
        def __init__(self, source, cited):
            self.source = source
            self.meta = {'cited_pages': cited}

    class _StubL1QA:
        """把 L1 的真实行为投影到桩上: 命中缓存就返回 source='redis',
        否则走"生成"分支并**写回缓存** —— 这样"清冷有没有生效"才真的被考到,
        而不是把答案写死成常量(写死的话, 清不清冷它都返回同一个 source)。"""

        def __init__(self, redis_client):
            self.redis_client = redis_client

        def answer_with_meta(self, question, session_id=None):
            key = _cache_key(question)
            if self.redis_client.get_data(key):
                return _StubL1Result('redis', [])
            self.redis_client.store[key] = {'answer': '桩答案', 'origin': 'l4_rag'}
            return _StubL1Result('rag', [114])

    l1_key = _cache_key('怎样加热座椅？')     # 必须与 check_l1_functional 里那句问句一致
    degraded_mark = '写端本次未被考到'

    # 绿(冷路径): 故意先塞一条"上次跑留下的"缓存 -> 函数应当把它删掉, 第一遍走冷路径。
    stub11 = _StubL1QA(_StubRedis([l1_key]))
    fake_app11 = types.ModuleType('app')
    fake_app11.qa_system = stub11
    saved_app11 = sys.modules.get('app')
    sys.modules['app'] = fake_app11
    buf11 = io.StringIO()
    try:
        with redirect_stdout(buf11):
            out11 = check_l1_functional()
    finally:
        if saved_app11 is None:
            del sys.modules['app']
        else:
            sys.modules['app'] = saved_app11
    printed11 = buf11.getvalue()
    assert out11['first_source'] != 'redis', \
        f'⑪ 清冷后第一遍仍是 redis({out11["first_source"]}) —— 那条 key 没被删掉, ' \
        f'写端根本不会被考到, 而结论句会照旧说"缓存通道是通的"'
    assert out11['pre_cached'] is True, \
        f'⑪ 清冷前它本来就在缓存里, 应当报出来(pre_cached), 实得 {out11["pre_cached"]}'
    assert out11['second_source'] == 'redis', \
        f'⑪ 第二遍应当命中刚写回去的缓存, 实得 {out11["second_source"]}'
    assert out11['first_pass_was_cold'] is True, f'⑪ first_pass_was_cold 应记 True'
    assert degraded_mark not in printed11, \
        f'⑪ 第一遍明明走了冷路径, 却打出了降级声明 —— 这条声明必须只在第一遍命中缓存时出现'

    # 红(降级): 让 delete 变成空操作, 造出"第一遍就命中"的情形 -> 必须打出降级声明。
    stub11b = _StubL1QA(_StubRedis([l1_key], delete_works=False))
    fake_app11b = types.ModuleType('app')
    fake_app11b.qa_system = stub11b
    sys.modules['app'] = fake_app11b
    buf11b = io.StringIO()
    try:
        with redirect_stdout(buf11b):
            out11b = check_l1_functional()
    finally:
        if saved_app11 is None:
            del sys.modules['app']
        else:
            sys.modules['app'] = saved_app11
    assert out11b['first_source'] == 'redis', \
        f'⑪ 替身没造出"第一遍就命中"的情形(实得 {out11b["first_source"]}), 这条断言失去了意义'
    assert degraded_mark in buf11b.getvalue(), \
        f'⑪ 第一遍就命中缓存, 却没打降级声明 —— 那句"缓存通道是通的"又变成了无条件结论'
    print(f'⑪ check_l1_functional: 清冷后第一遍走冷路径; 第一遍仍命中时打出降级声明')

    # ⑫ 基线落盘这一对(F6 + F7 + M4): ①目标文件已存在 -> main() 拒绝开工**且不碰旧文件**;
    #    ②写到一半崩 -> 旧文件仍在且仍是合法 JSON; ③段名不在 _ALL_SECTIONS 里当场炸;
    #    ④汇总印的 meta 就是刚写进文件的那一份。
    #    为什么非钉不可: 基线 JSON 是**唯一不可替代**的落盘物(逐条付费换来的数),
    #    而"覆盖"与"写坏"都是**静默**的 —— 出了事连上一次的数都拿不回来。
    #    零出网零花费: 全在临时目录, 用 `--only intent`(只读本地文件, 不连任何真服务);
    #    就算闸门失效, 这条路也跑不出任何付费调用。
    tmp4 = tempfile.mkdtemp(prefix='eval_rag_selfcheck_')
    EVAL_DIR = tmp4
    saved_argv = sys.argv
    try:
        target = baseline_path('intent')
        old_bytes = b'{"meta": {}, "metrics": {}}\n'      # 冒充"上一次跑留下的基线"
        with open(target, 'wb') as f:
            f.write(old_bytes)

        sys.argv = ['eval_rag.py', '--only', 'intent']
        buf12 = io.StringIO()
        with redirect_stdout(buf12):
            rc = main()
        assert rc == 2, \
            f'⑫ 目标基线已存在时 main() 应拒绝开工(返回 2), 实得 {rc} —— ' \
            f'不拦的话, 这一次的结果会**静默覆盖**上一份(可能是花钱买来、不可重跑的)'
        assert '已存在' in buf12.getvalue(), '⑫ 拒绝开工了, 却没印出让用户怎么办'
        with open(target, 'rb') as f:
            assert f.read() == old_bytes, '⑫ 拒绝开工了却还是动了旧基线文件'

        # ② 写到一半崩: 拿一个 **json 序列化不了的 payload**(集合)当桩 ——
        #    json.dump 会先写出一截字节再抛 TypeError, 正是"死在写窗口里"的形状。
        with redirect_stdout(io.StringIO()):
            try:
                save_baseline({'L4_multimodal': {'n_valid': 1, 'bad': {1, 2}}}, 'intent')
            except TypeError:
                pass
            else:
                raise AssertionError('⑫ 序列化不了的 payload 居然写成功了 —— 这个桩没起作用')
            with open(target, 'rb') as f:
                assert f.read() == old_bytes, \
                    '⑫ 写失败后旧基线被改动了 —— 唯一不可替代的那份产物就毁在这一刻'
            with open(target, encoding='utf-8') as f:
                json.load(f)                                   # 仍是合法 JSON
            assert not os.path.exists(target + '.tmp'), '⑫ 写失败后留下了半截临时文件'

        # ③ 段名单闸(F7): 段名不在 _ALL_SECTIONS 里必须当场炸。
        with redirect_stdout(io.StringIO()):
            try:
                save_baseline({'L4_no_such_section': {'n_valid': 1}}, 'intent')
            except AssertionError as e:
                assert '_ALL_SECTIONS' in str(e), f'⑫ 炸了, 但炸出来的不是段名单那道闸: {e}'
            else:
                raise AssertionError(
                    '⑫ 段名不在 _ALL_SECTIONS 里却没炸 —— 那段会在 sections_run 与 '
                    'sections_not_run 里**同时消失**, 报告看着完整而一整段没了')

        # ④ 汇总必须用**传进来的** meta, 不许当场重算一份(M4)。
        #    用哨兵字符串判定: 真重算的话屏幕上永远看不到这个值。
        buf12d = io.StringIO()
        with redirect_stdout(buf12d):
            print_summary({}, {'time': 'SENTINEL-TIME', 'git_commit': 'SENTINEL-COMMIT',
                               'dataset_fingerprint': {}, 'judge_model': 'SENTINEL-MODEL',
                               'embedding': 'SENTINEL-EMB', 'valid_samples': {},
                               'sections_run': [], 'sections_not_run': []})
        assert 'SENTINEL-TIME' in buf12d.getvalue(), \
            '⑫ print_summary 没用传进来的 meta(或是自己又 collect_meta 了一遍)—— ' \
            '屏幕上印的头部会与文件里的 meta 不是同一份'

        # ⑤ **`collect_meta` 真把有效样本数捞出来了**(复审 M2)。
        #    为什么非钉不可: 上面 ④ 传的是**手写** meta(它自己就带着 `'valid_samples': {}`),
        #    ② 虽然让 collect_meta 跑过一次却**不断言它的输出** —— 于是"collect_meta 里那段
        #    过滤写坏了"这件事**没有任何断言看得见**: 自检照样全绿, 而将来每一份基线的
        #    `valid_samples` 恒为 `{}`, F4 白做且无人报。判据必须是"捞到了具体的数",
        #    不是"这个键存在"(空 dict 也满足"存在")。
        meta12 = collect_meta({'L4_multimodal': {'n_valid': 71},
                               'gen_text': {'n': 96, 'n_valid': 96, 'faithfulness_valid': 94}})
        assert meta12['valid_samples'].get('L4_multimodal') == {'n_valid': 71}, \
            f'⑫ collect_meta 没把 n_valid 捞进 valid_samples: {meta12["valid_samples"]}'
        assert meta12['valid_samples'].get('gen_text') == \
            {'n_valid': 96, 'faithfulness_valid': 94}, \
            f'⑫ collect_meta 没把逐指标的 *_valid 一并捞出来: {meta12["valid_samples"]}'
        # 反面: 没有 n_valid/*_valid 的段不该被凭空塞进去(否则 valid_samples 会变噪声)。
        assert 'L4_refuse' not in collect_meta({'L4_refuse': {'n_in': 30}})['valid_samples'], \
            '⑫ 没有 n_valid/*_valid 的段被塞进了 valid_samples'
        # ⑥ git 状态: ①**冻结值优先**于现取的; ②`git_dirty` 真的跟着工作区状态变。
        #    为什么非钉不可: `git_commit` 标示"这批数出自哪一版"; 若每次都现取,
        #    **跨提交的长跑会把数字挂在一个它从未执行过的版本上**, 方向恰好是"看起来更可信"
        #    (实测发生过: 那次 --only retrieval 跑着的时候才提交)。
        #    ①用哨兵值验"冻结赢"; ②拿一个**临时仓库**验判据本身 —— 绝不去碰本项目仓库。
        live = snapshot_git_state()
        saved_frozen = me._FROZEN_GIT
        me._FROZEN_GIT = {'commit': 'SENTINEL-COMMIT', 'dirty': 'SENTINEL-DIRTY'}
        try:
            meta12b = collect_meta({})
        finally:
            me._FROZEN_GIT = saved_frozen
        assert meta12b['git_commit'] == 'SENTINEL-COMMIT', \
            f'⑫ collect_meta 没用冻结的 commit(实得 {meta12b["git_commit"]!r})—— ' \
            f'长跑跨提交时, 数字会被挂在一个它从未执行过的版本上'
        assert live['commit'] != 'SENTINEL-COMMIT', \
            '⑫ 现取的 commit 与哨兵值相同 —— 这条断言就没有区分力了'
        assert meta12b['git_dirty'] == 'SENTINEL-DIRTY', \
            f'⑫ collect_meta 没用冻结的 git_dirty(实得 {meta12b["git_dirty"]!r})'

        tmpgit = tempfile.mkdtemp(prefix='eval_rag_selfcheck_git_')
        try:
            subprocess.run(['git', 'init', '-q'], cwd=tmpgit, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            assert snapshot_git_state(tmpgit)['dirty'] is False, \
                '⑫ 空仓库(无任何改动)应判"干净"'
            with open(os.path.join(tmpgit, 'x.txt'), 'w', encoding='utf-8') as f:
                f.write('x')
            assert snapshot_git_state(tmpgit)['dirty'] is True, \
                '⑫ 有未跟踪文件的工作区应判"不干净" —— 判反了的话, "跑的那一版没有哈希' \
                '能指代"这件事就永远不会自己显形(只能靠后来的人去比时间戳)'
        finally:
            shutil.rmtree(tmpgit, ignore_errors=True)
        assert not os.path.exists(tmpgit), f'⑫ 临时 git 仓库没被清掉: {tmpgit}'
        print('⑫ 基线落盘: 已存在则拒绝开工且不碰旧文件 / 写失败旧文件仍是合法 JSON / '
              '段名单闸会响 / 汇总用的是写进文件的那份 meta / collect_meta 真捞到有效样本数 / '
              'git_commit 用的是开跑前冻结的那一版(且 git_dirty 跟工作区状态走)')
    finally:
        sys.argv = saved_argv
        EVAL_DIR = saved_dir
        shutil.rmtree(tmp4, ignore_errors=True)

    assert not os.path.exists(tmp4), f'⑫ 临时目录没被清掉: {tmp4}'

    # ⑬ F1: 单通道多模态的**原始证据**必须真的落成文件(multimodal_raw.json)。
    #    为什么非钉不可: 这是本轮最高价值的一条, 而它的失效**全是静默的** ——
    #    少写一列、gold 写成 str、落在一个能被别的路径覆盖的名字上…… 没有一个会报错,
    #    而落盘一旦不对, R@k/MRR 就永远回不到"能从盘上产物重算"。
    #    零出网零花费: 桩掉的是**外部重模型**(PageRetriever),
    #    跑的是**真的** eval_multimodal 本体, 读的也是**真的** eval_set。
    #    (替身只能替外部依赖 —— 替了被测函数本身, 这条自检就什么都没验。)
    class _StubPageRetriever:
        def __init__(self, load_on_init=False):
            pass

        def image_leg(self, question, k=MM_K):
            # 全序固定为 1..k: 落盘的 ranked 必须与它逐位相同, 才说明落的是全序不是前几名。
            #   ⚠️ 桩必须回**真形状** (SOURCE_ID, page, score) —— 写 (page, score) 的话
            #   page_to_int 把 page 当成 source_id, 全被"别书"剔除, ranked 变空, 断言立刻假红。
            return [(SOURCE_ID, p, 1.0) for p in range(1, k + 1)]

    tmp5 = tempfile.mkdtemp(prefix='eval_rag_selfcheck_')
    saved_dir5 = EVAL_DIR
    EVAL_DIR = tmp5
    saved_mod = sys.modules.get('rag_qa.core.image_leg')
    fake_image_leg = types.ModuleType('rag_qa.core.image_leg')
    fake_image_leg.PageRetriever = _StubPageRetriever
    sys.modules['rag_qa.core.image_leg'] = fake_image_leg
    try:
        with redirect_stdout(io.StringIO()):
            m_multi = eval_multimodal()

        raw_path = os.path.join(tmp5, 'multimodal_raw.json')
        assert os.path.isfile(raw_path), f'⑬ 多模态检索的原始证据没落盘: {raw_path}'

        with open(raw_path, encoding='utf-8') as f:
            dumped = json.load(f)

        # 期望的 id 序列直接从**真**数据集算, 不写死条数: 数据集变了这条自检不该跟着假红。
        with open(EVAL_SET, encoding='utf-8') as f:
            expect_ids = [r['id'] for r in json.load(f) if r['gold_pages']]

        assert [d['id'] for d in dumped] == expect_ids, \
            '⑬ 落盘的 id 序列与真数据集(有 gold 的那批)对不上'
        assert all(set(d) == {'id', 'gold', 'ranked'} for d in dumped), \
            f'⑬ 每条应当只有 id/gold/ranked 三个键, 实得 {sorted(dumped[0])}'
        assert all(d['ranked'] == list(range(1, MM_K + 1)) for d in dumped), \
            '⑬ 落盘的 ranked 不是检索返回的**全序**'
        assert all(isinstance(d['gold'], list) and all(isinstance(p, int) for p in d['gold'])
                   for d in dumped), \
            '⑬ 的 gold 应是 list[int](gold_pages 的本来的形态)'
        # 干净数据的交叉核对: 桩恒返回 1..k, 所以每条的首个 gold 页若落在 k 内, R@k 必然是 1。
        with open(EVAL_SET, encoding='utf-8') as f:
            expect_n_valid = sum(
                1 for r in json.load(f)
                if r['gold_pages'] and len(r['gold_pages']) <= GOLD_MAX_PAGES)
        assert m_multi['n_valid'] == expect_n_valid, \
            f'⑬ 的 n_valid 与"gold 非空且不超 {GOLD_MAX_PAGES} 页"的条数对不上'

        # **带 limit 的冒烟跑必须换名, 不许顶掉全量那份证据**(复审 M4②)。
        #   为什么非钉不可: 这不报错、也不影响任何指标 —— 只是盘上那份"能算出 R@k"的
        #   原始证据被 2 条顶掉, 而文件本身不自描述, 事后**没人分得出来**。
        #   判据两条缺一不可: ①limit 版落到**另一个**文件; ②原文件**内容一字未变**。
        with redirect_stdout(io.StringIO()):
            eval_multimodal(limit=2)
        limited_path = os.path.join(tmp5, 'multimodal_raw_limit2.json')
        assert os.path.isfile(limited_path), \
            f'⑬ 带 limit 的冒烟跑没换文件名, 落盘位置仍是 {os.path.join(tmp5, "multimodal_raw.json")}'
        with open(limited_path, encoding='utf-8') as f:
            assert len(json.load(f)) == 2, '⑬ limit=2 的产物里应恰好 2 条'
        with open(raw_path, encoding='utf-8') as f:
            assert len(json.load(f)) == len(expect_ids), \
                '⑬ 带 limit 的冒烟跑把**全量**那份证据顶掉了 —— 事后没人分得出这份证据' \
                '是不是被截过(记忆: hardcoded-output-paths-overwrite-silently)'
        print(f'⑬ F1 原始证据落盘: multimodal_raw({len(dumped)} 条, gold=list[int]); '
              f'limit=2 冒烟跑另落 *_limit2.json, 不动全量那份')
    finally:
        EVAL_DIR = saved_dir5
        if saved_mod is None:
            # 桩模块必须撤掉: 留在 sys.modules 里, **同一进程后续**的真跑会拿到它 —— 那时
            #   "检索"变成恒返回 1..k, 数字全是假的而且不会报错。
            sys.modules.pop('rag_qa.core.image_leg', None)
        else:
            sys.modules['rag_qa.core.image_leg'] = saved_mod
        shutil.rmtree(tmp5, ignore_errors=True)

    assert not os.path.exists(tmp5), f'⑬ 临时目录没被清掉: {tmp5}'

    # ⑭ F3 + M5: 端到端 ①把"按层拆分"印出来; ②P95 取的是最近秩那个样本(不是偏大的那个)。
    #    为什么非钉不可:
    #      ①"命中率 75%"在报告里看不出它是"rag 15/15 + mysql 0/5"还是"rag 10/15 + mysql 5/5",
    #        而这两件事的结论完全相反 —— 后者是检索问题, 前者是"那一层根本没有页码"。
    #      ②`int(0.95n)` 在 0.95n 为整数时偏大一格, n=20 直接退化成 max。当前 n=30 同值,
    #        所以它是个**潜伏**问题: 谁改了 n 就中招, 而 P95 变大看着像"延迟变差了"。
    #    零出网零花费: 桩 app + 桩 redis + 假数据集(20 条), n=20, 不连任何真服务。
    class _StubE2EMultiLayers:
        """前 15 条 source='rag' 且引用了 gold(命中), 后 5 条 source='mysql' 且 cited=[]。"""

        def __init__(self):
            self.redis_client = _StubRedis([])
            self.calls = 0

        def answer_with_meta(self, question, session_id=None):
            self.calls += 1
            if self.calls <= 15:
                return _StubE2EResult('rag', [1])
            return _StubE2EResult('mysql', [])

    class _FakeTime:
        """
        只替 perf_counter(其余属性转发真 time 模块) —— 让 20 条延迟精确可控。

        真模块必须**构造时传进来**: 类体里直接写 `time` 会解析到模块全局的 `time`,
        而它下一行就被换成这个实例本身 ⇒ `__getattr__` 无限递归(实测踩过, RecursionError)。
        """

        def __init__(self, values, real):
            self._values = list(values)
            self._real = real

        def perf_counter(self):
            return self._values.pop(0)

        def __getattr__(self, name):
            return getattr(self._real, name)

    tmp6 = tempfile.mkdtemp(prefix='eval_rag_selfcheck_')
    saved_dir6 = EVAL_DIR
    saved_set6 = EVAL_SET
    saved_time = me.time
    EVAL_DIR = tmp6
    stub_rows20 = [{'id': f'r{i:03d}', 'question': f'桩问题{i}', 'gold_pages': [1]}
                   for i in range(1, 21)]
    with open(os.path.join(tmp6, 'eval_set.json'), 'w', encoding='utf-8') as f:
        json.dump(stub_rows20, f, ensure_ascii=False)
    EVAL_SET = os.path.join(tmp6, 'eval_set.json')

    fake_app14 = types.ModuleType('app')
    fake_app14.qa_system = _StubE2EMultiLayers()
    saved_app14 = sys.modules.get('app')
    sys.modules['app'] = fake_app14
    # 延迟脚本: perf_counter 每次迭代被调两次(t0 与 t0+elapsed), 所以喂的是前缀和序列,
    #   这样 20 条各自的 elapsed 恰好是 1,2,...,20。
    prefix, scripted = 0, []
    for e in range(1, 21):
        scripted.extend([prefix, prefix + e])
        prefix += e
    me.time = _FakeTime(scripted, time)      # 第二个参数是**当时还没被换掉**的真 time 模块
    buf14 = io.StringIO()
    try:
        with redirect_stdout(buf14):
            out14 = eval_e2e(n=20)
    finally:
        me.time = saved_time
        EVAL_SET = saved_set6
        EVAL_DIR = saved_dir6
        if saved_app14 is None:
            sys.modules.pop('app', None)
        else:
            sys.modules['app'] = saved_app14
        shutil.rmtree(tmp6, ignore_errors=True)

    assert not os.path.exists(tmp6), f'⑭ 临时目录没被清掉: {tmp6}'
    printed14 = buf14.getvalue()
    assert out14['layers'] == {'rag': 15, 'mysql': 5}, f'⑭ 分层计数错了: {out14["layers"]}'
    assert out14['layer_hits'] == {'rag': 15, 'mysql': 0}, \
        f'⑭ 按层命中数错了: {out14["layer_hits"]}'
    assert '按层拆分' in printed14, '⑭ 没印按层拆分 —— 混合总体又只有一个平均数了'
    assert 'mysql 0/5' in printed14 and 'rag 15/15' in printed14, \
        f'⑭ 按层拆分印出来了但数字不对'
    assert '结构性为 0' in printed14, \
        '⑭ mysql 那一层没标注"结构性为 0" —— 读的人会以为那一层"答错了页"'
    # M5: 延迟恰为 1..20, 最近秩 P95 = 第 ceil(0.95*20)=19 个(1 基) = 19;
    #   旧写法 int(20*0.95)=19 取下标 19 = 第 20 个 = 20(退化成 max)。
    assert out14['p95'] == 19, \
        f'⑭ P95 取了 {out14["p95"]}, 应取第 19 个样本(19) —— ' \
        f'取到 20 说明用了 int(0.95n), n 是 20 的倍数时会静默退化成 max'
    assert out14['p50'] == 11, f'⑭ P50 不该被动到, 实得 {out14["p50"]}'
    print(f'⑭ F3+M5: 按层拆分(rag 15/15, mysql 0/5 标注结构性 0)印出; '
          f'P95={out14["p95"]} 取的是最近秩那个样本')

    # ⑮ report_recall 的**池深摘要**(复审第 4 轮) —— 深度已知的桩 rows, 断言印出来的数与
    #    返回的键**恰好**是期望值。
    #    为什么非钉不可: `Recall@k` 的分母是"检索返回的那个池子", 而池子未必有 k 那么深
    #    (按页去重的必然结果, 机理与实测见 report_recall 的 docstring) —— 实测 **66/71 条**
    #    的 R@15 名不副实。这条摘要就是把那个构成摆出来; **它算错了不会让任何指标变红**,
    #    只会让读者继续误读 —— 正是"头条数字藏着构成"那一类(F3 同形)。
    stub_depth_rows = [
        {'ranked': [1, 2, 3, 4], 'gold': [1]},                  # 池深 4
        {'ranked': [1, 2, 3, 4, 5, 6, 7, 8], 'gold': [1]},      # 池深 8
        {'ranked': list(range(1, 13)), 'gold': [1]},            # 池深 12
        {'ranked': list(range(1, 16)), 'gold': [1]},            # 池深 15
    ]
    buf15 = io.StringIO()
    with redirect_stdout(buf15):
        m15 = report_recall(stub_depth_rows, [1, 5, 15], '自检桩-池深')
    assert m15['depth_min'] == 4, f'⑮ depth_min 应为 4, 实得 {m15["depth_min"]}'
    assert m15['depth_max'] == 15, f'⑮ depth_max 应为 15, 实得 {m15["depth_max"]}'
    # 中位数(10.0)与均值(9.75)**故意不相等**: 桩就是这么挑的, 否则"取错统计量"这条测不出来。
    assert m15['depth_median'] == 10.0, \
        f'⑮ 池深中位数应为 10.0(4/8/12/15 的中位), 实得 {m15["depth_median"]}(这四个数的均值是 9.75)'
    # **恰好 3 不是 4**: 深度是 4/8/12/15, 严格小于 15 的只有前三条 —— 第四条**等于** 15,
    #   不算"池子不够深"。这条期望值我第一版写错成 4, 是自检当场把它打红的。
    assert m15['n_pool_lt_kmax'] == 3, \
        f'⑮ 池深 < 15 的行数应为 3(4/8/12; 第 4 条恰等于 15 不算), 实得 {m15["n_pool_lt_kmax"]}'
    assert '池深' in buf15.getvalue(), \
        '⑮ 池深摘要没印出来 —— 读者就看不出 R@15 对哪些行名不副实'
    print(f'⑮ report_recall 池深摘要: min {m15["depth_min"]} / 中位 {m15["depth_median"]:g} / '
          f'max {m15["depth_max"]}, 池深 < 15 的行数 {m15["n_pool_lt_kmax"]} —— 全部与桩相符')

    print('自检: ①~⑮ 全过')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selfcheck', action='store_true', help='只跑纯函数自检')
    ap.add_argument('--build-set', action='store_true', help='重建评估数据集')
    ap.add_argument('--only', choices=['retrieval', 'l2', 'refuse', 'intent',
                                       'generation', 'e2e'],
                    help='只跑某一段; 不给则全跑')
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    if args.build_set:
        return build_eval_set()

    # ---- F6 防呆闸: 目标基线文件已存在就**别开工** ----
    # 【为什么"存在就拒绝"而不是改名走人】`save_baseline` 是**逐段覆盖同一个文件**的
    #   (崩在段中间时那份文件就是唯一的部分报告), 所以不能写成"文件在就自动换个名" ——
    #   那样一次全量跑会散成好几个文件, "到此为止跑成的全部内容"就不在一个地方了。
    #   命名规则也不动(裁决 2: 全量 YYYYMMDD, --only 带后缀)。
    # 【为什么要在这儿查而不是等到落盘时】实测教训(记忆 hardcoded-output-paths-overwrite-silently):
    #   跑完几十分钟甚至几小时之后才发现同名, 那时上一次的结果**已经被覆盖掉**了。
    #   而且这儿的检查在任何一段开跑**之前** —— 一个付费调用都不会花出去。
    path = baseline_path(args.only)
    if os.path.exists(path):
        print(f'❌ 基线文件已存在, 拒绝开工: {path}')
        print('   继续跑会用本次结果**覆盖**掉它, 而上一份可能是花钱买来、不可重跑的数。')
        print('   请二选一: ①把旧文件改名/移到别处留档; ②确认它没用后删掉, 再重跑。')
        return 2

    # ---- 冻结"本次跑的到底是哪一版" ----
    # 必须在**任何一段开跑之前**取一次: `collect_meta` 是逐段落盘时各调一次的, 现取的话
    # 记的是"落盘那一刻的 HEAD"而不是"真正执行的那一版" —— 长跑跨提交就会把数字挂在一个
    # 它从未执行过的版本上(实测发生过, 见 snapshot_git_state 的 docstring)。
    global _FROZEN_GIT
    _FROZEN_GIT = snapshot_git_state()

    metrics = {}
    if args.only in (None, 'retrieval'):
        metrics['L4_multimodal'] = eval_multimodal()
        save_baseline(metrics, args.only)      # 每段之后落一次: 累计覆盖同一个文件
    if args.only == 'l2':
        # 全量跑**刻意不含 L2**(100 次付费调用, 属用户决策) —— 别改成 `in (None, 'l2')`:
        #   那会让"只买一次"的那次全量悄悄多花一笔钱。未跑这件事由 sections_not_run 报出来。
        metrics['L2_baseline'] = eval_l2(rewrite=False)
        metrics['L2_rewrite'] = eval_l2(rewrite=True)
        save_baseline(metrics, args.only)
    if args.only in (None, 'refuse'):
        metrics['L4_refuse'] = eval_refuse()
        save_baseline(metrics, args.only)
    if args.only in (None, 'intent'):
        metrics['L3_intent'] = eval_intent()
        save_baseline(metrics, args.only)
    if args.only in (None, 'generation'):
        metrics['gen_text'] = eval_generation_text()
        save_baseline(metrics, args.only)
    if args.only in (None, 'e2e'):
        metrics['e2e'] = eval_e2e()            # 默认 n=30
        metrics['l1_functional'] = check_l1_functional()
        save_baseline(metrics, args.only)

    # 各段**不**各自 try/except: 吞异常 = 让"这段没测成"静默变成"这段没数据"(本项目禁止
    #   静默吞异常)。就让它炸 —— 前面各段已经逐段落过盘, 那些付费数字丢不掉。
    # 末尾这次落盘**先写文件、后打印汇总**, 并把写进去的那份 meta 交给 print_summary ——
    #   屏幕上的头部与文件里的 meta 是同一份, print_summary 不再重算。
    payload = save_baseline(metrics, args.only)     # 末尾保留一次: 钉住最终态
    print_summary(metrics, payload['meta'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
