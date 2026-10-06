# 离线生成 L3 意图识别的训练数据(汽车域, 一次性)。
#   正类从手册 PDF 逐页造题(不是 jpkb —— L3 下沉到 L4 查手册, 正类定义是"手册答得了的问题");
#   负类掺两类硬负样本(旧域 IT 题 / 提到车但手册答不了的), 它们真正决定边界质量。
# 跑法: PYTHONPATH=. python rag_qa/classify_data/gen_intent_dataset.py [--rebuild 忽略缓存]
# 产物: rag_qa/classify_data/car_intent_5000.json (JSON Lines: query / label / page)
# 前提: .env 有可用的 DASHSCOPE_API_KEY
#
# 可断点续跑: 每页结果单独落盘到 _gen_cache/, 重跑时已有页直接读缓存, 不再烧 API 调用。

import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# 路径引导: 只插项目根到 sys.path(家规 A8)。本文件比项目根浅 3 层。
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from base.logger import logger
from rag_qa.core.llm_client import ERROR_PREFIX, get_llm_client

# 项目内绝对路径, 不受运行目录影响(与 train_classifier.py 同一套约定)。
RAG_QA_DIR = os.path.dirname(_HERE)

# L4 真正检索的那份 PDF。**别换成 ../DATA/AudiManuals/ 下那 10 份** —— 它们没入库,
#   car_mm / car_text 里一条都没有, 从它们造题会造出"手册答得了但检索库不存在"的问题。
DEFAULT_PDF = os.path.abspath(
    os.path.join(_PROJECT_ROOT, 'rag_qa', 'data', 'car_data', 'train_a.pdf'))

OUT_PATH = os.path.join(_HERE, 'car_intent_5000.json')
CACHE_DIR = os.path.join(_HERE, '_gen_cache')

# 每类目标条数(提示词模块.txt 要求: 5000 条, 两类均衡)。
TARGET_PER_CLASS = 2500

# 每页造几个问题。可用页约 290, 9 x 290 ≈ 2600, 去重后留够 2500。
Q_PER_PAGE = 9

# 过滤太短的页(封面/空白/插图页)。实测 354 页里 >=120 字的有 303 页。
MIN_PAGE_CHARS = 120

# 目录页特征: 大量点线引导符。这类页造"第 11 页讲了什么"对分类器是噪声, 剔掉。
TOC_DOTS = re.compile(r'\.{5,}')

# 前言区判据(实测 p0~13: 封面欢迎辞 / 目录 / 前言)。只按字数过滤时封面 244 字欢迎辞
#   会被逼造出"本公司将不断改进意思是?"这类反问元问题, 教坏分类器。取"开头 30 字出现标题词"。
FRONT_MATTER_WORDS = ('欢迎', '目录', '前言')


def _is_front_matter(text):
    """前言/目录/封面判据: 标题词出现在页首。"""
    return any(word in text[:30] for word in FRONT_MATTER_WORDS)


def _is_toc(text):
    """目录页判据: 点线引导符出现 5 次以上(实测目录页有几十处)。"""
    return len(TOC_DOTS.findall(text)) >= 5


def iter_usable_pages(pdf_path, limit=None):
    """
    函数功能: 逐页抽字 -> 过滤 -> yield (page_no, text)。
    fitz 直接抽文字层('text' 模式), 0 基页号。检索页图为多模态那条路, 造题看正文足够。
    """
    import fitz
    doc = fitz.open(pdf_path)
    try:
        for page_no in range(doc.page_count):
            text = doc[page_no].get_text('text').strip()
            if len(text) < MIN_PAGE_CHARS or _is_toc(text) or _is_front_matter(text):
                continue
            # 一页最多送 1800 字: 手册最长页 4408 字, 全送挤掉输出空间且后段多是页脚噪声
            yield page_no, text[:1800]
            if limit is not None and page_no >= limit:
                return
    finally:
        doc.close()


# ============================================================
# 正类: 逐页造"这页答得了的问题"
# ============================================================
POSITIVE_PROMPT = '''你是汽车售后客服系统的语料工程师。下面是一份汽车使用手册的某一页的正文。

请生成 {n} 个**真实车主会问出口的问题**, 要求这些问题**必须依靠这一页的内容才能回答**。

问法要混着来, 四种各占一些:
1. 口语化故障描述(如「车打不着火了」「仪表盘亮黄灯是什么意思」)
2. 功能怎么用(如「定速巡航怎么打开」「座椅怎么调」)
3. 操作步骤(如「怎么换备胎」「怎么设置自动落锁」)
4. 专有名词询问(如「HUD 是什么」「胎压监测怎么复位」)

硬性要求:
- 每条必须能被这一页的内容回答; 这一页没提到的功能或故障一律不要问
- 要像车主的口吻, 不要写成教科书标题、不要写成"XX 功能说明"
- 不要出现"本页""根据手册""第几页"这类字眼
- 不要问通用常识(如"1+1等于几""Python 怎么写"), 那些不是手册能答的
- 每条不超过 30 字, 长度参差一些, 不要都一个句式
- 各条之间不要重复

只输出 JSON 数组, 不要任何解释文字:
["问题1", "问题2", ...]

--- 手册正文 ---
{text}'''


def _extract_json_array(raw):
    """
    函数功能: 从模型返回里抠出 JSON 数组。
    模型常包 ```json 或写句"好的..." -> 退化到取第一个 [ 到最后一个 ] 再解析一次。
    返回 list; 解析不出返回 None(调用方据此判失败并重试)。
    """
    if not raw or raw.startswith(ERROR_PREFIX):
        return None
    candidates = [raw.strip()]
    start, end = raw.find('['), raw.rfind(']')
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])
    for text in candidates:
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def _clean_question(item):
    """
    函数功能: 清洗单条问题 -> 合法字符串 或 None。
    模型会夹带非字符串元素、把问题写成 {"query": ...} 字典、或带"问题1:"序号前缀。
    """
    if isinstance(item, dict):
        item = item.get('query') or item.get('question') or ''
    if not isinstance(item, str):
        return None
    q = re.sub(r'^\s*(问题\s*\d+|\d+)[.、:：)）]\s*', '', item).strip()
    q = q.strip('"\' ')
    if not (4 <= len(q) <= 60):
        return None
    return q


def gen_positive_page(client, page_no, text, retries=3):
    """函数功能: 为一页生成问题列表(带重试)。返回 (page_no, list[str]); 彻底失败 (page_no, [])。"""
    prompt = POSITIVE_PROMPT.format(n=Q_PER_PAGE, text=text)
    for attempt in range(retries):
        # 温度给高: 要问法多样不要稳(与 L3 直答要稳的 0.1 相反)
        raw = client.chat_text(prompt, temperature=0.9)
        arr = _extract_json_array(raw)
        if arr is not None:
            cleaned = [q for q in (_clean_question(x) for x in arr) if q]
            if cleaned:
                return page_no, cleaned
        logger.warning(f'第 {page_no} 页解析失败(第 {attempt + 1}/{retries} 次)')
    return page_no, []


def gen_positives(client, pdf_path, rebuild=False):
    """
    函数功能: 全部页 -> 问题列表。每页单独落盘缓存, 中断后重跑只补缺的页。
    :return: list[dict], 形如 {'query':..., 'label': '专业咨询', 'page': page_no}
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 先划出要处理的页; 已有缓存的页跳过(除非 --rebuild)。
    pages = []
    cached = {}
    for page_no, text in iter_usable_pages(pdf_path):
        cache_file = os.path.join(CACHE_DIR, f'p{page_no:03d}.json')
        if not rebuild and os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached[page_no] = json.load(f)
        else:
            pages.append((page_no, text, cache_file))

    logger.info(f'正类: 共 {len(pages) + len(cached)} 页可用, '
                f'缓存命中 {len(cached)} 页, 本次需出网 {len(pages)} 页')
    print(f'正类: 可用页 {len(pages) + len(cached)}, 缓存命中 {len(cached)}, 待生成 {len(pages)}')

    # 并发跑剩下的页。并发压到 4: DashScope 有 QPM 限制, 开太大全在重试上耗时间。
    results = dict(cached)
    if pages:
        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(gen_positive_page, client, pno, txt): (pno, cf)
                for pno, txt, cf in pages
            }
            for fut in as_completed(futures):
                pno, cache_file = futures[fut]
                try:
                    _, questions = fut.result()
                except Exception as e:
                    logger.error(f'第 {pno} 页生成异常: {e}')
                    questions = []
                results[pno] = questions
                # 立刻落盘: 下一页失败不该把这一页的成果也搭进去
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(questions, f, ensure_ascii=False)
                done += 1
                if done % 20 == 0:
                    print(f'  正类进度 {done}/{len(pages)}', flush=True)

    rows = [
        {'query': q, 'label': '专业咨询', 'page': pno}
        for pno in sorted(results) for q in results[pno]
    ]
    failed = sum(1 for pno in results if not results[pno])
    logger.info(f'正类: {len(rows)} 条(其中 {failed} 页产出为空)')
    return rows


# ============================================================
# 负类: 通用问题 + 两类硬负样本
# ============================================================
# (主题名, 条数, 给模型的说明)。前 10 个普通负类, 最后两个硬负样本(IT 培训 / 提车但手册
# 答不了)各 250 条, 占 raw ~17%。它们也走随机抽样(约去 1/6), 不额外保护: 抽样无偏, 留下的
#   ~420 条已够顶住边界, 为它们开"必留"旁路只会多一种配平口径。
NEGATIVE_TOPICS = [
    ('数学与逻辑', 250, '数学计算、逻辑推理、单位换算'),
    ('编程与代码', 250, '写代码、报错排查、编程语言语法'),
    ('计算机通识', 250, '操作系统、网络、软件使用的通用概念'),
    ('生活常识与健康', 250, '饮食、医疗、运动、家居等日常问题'),
    ('天气时事与娱乐', 250, '天气、新闻、影视、音乐、闲聊'),
    ('法律金融生活事务', 250, '租房、合同、社保、银行、保险等'),
    ('考试与升学', 250, '考研、四六级、公务员、职业资格考试'),
    ('其他领域专业咨询', 250, '医疗、法律、教育、心理等非汽车领域的专业问题'),
    ('情感与闲聊', 250,
     '心情倾诉、吐槽、求安慰、讲故事、没有明确信息需求的日常对话'),
    ('体育与旅行', 250,
     '体育赛事、健身锻炼、旅游攻略、景点推荐、行程规划(与汽车无关的出行)'),
    ('IT职业培训咨询', 250,
     'IT 培训机构的课程咨询: 学费多少、课程大纲、师资力量、学习周期、'
     '校区地址、就业情况、报名优惠(如 Java / Python / 前端 / 测试 / 大数据等方向)。'
     '**这一组非常重要**, 它们在过去被判成"专业咨询", 现在必须判成"通用知识"'),
    ('提到汽车但手册答不了', 250,
     '虽然和汽车相关, 但一本车主使用手册回答不了的问题: 车型对比选买建议'
     '(如"卡罗拉和朗逸哪个好")、汽车行业销量与新闻、二手车交易与过户、'
     '汽车品牌历史、车企业绩与股价、驾照考试技巧'),
]

NEGATIVE_PROMPT = '''你在为一个**汽车售后问答系统**构造意图分类器的负样本。

这个系统的知识库**只有一本车主使用手册**。判据只有一条:
- 手册内容能回答的问题 -> 「专业咨询」
- 其余一律 -> 「通用知识」

请生成 {n} 条属于「通用知识」的用户提问, 主题范围: {topic}

要求:
- 像真人随口问的话, 不要写成教科书标题
- 长度参差一些(6~30 字), 不要都是一个句式
- 各条之间不要重复, 不要只是换个名词的同一句话
- 不要问任何"需要查车主手册才能回答"的问题(那是正类, 不是你要生成的)

只输出 JSON 数组, 不要任何解释文字:
["问题1", "问题2", ...]'''


def gen_negative_topic(client, topic, count, desc, retries=3):
    """
    函数功能: 为一个主题批量造负样本。
    一次要 50 条而非逐条问 —— 负类没"必须贴某页"的约束, 批量更省调用,
    多样性靠主题切分(11 个主题)而非调用次数。
    """
    out = []
    batch = 50
    while len(out) < count:
        prompt = NEGATIVE_PROMPT.format(n=batch, topic=desc)
        got = []
        for attempt in range(retries):
            arr = _extract_json_array(client.chat_text(prompt, temperature=1.0))
            if arr is not None:
                got = [q for q in (_clean_question(x) for x in arr) if q]
                if got:
                    break
            logger.warning(f'负类[{topic}] 第 {attempt + 1}/{retries} 次解析失败(批 {batch} 条)')
        if got:
            out.extend(got)
            continue
        # 整批失败 -> **批大小减半再试**, 别就地放弃: 方括号类问题(如 `list[int]`)长输出
        #   易截断, JSON 解析必炸; 缩小批 = 缩短输出 = 更容易解析成功, 比整类放弃划算。
        if batch <= 10:
            logger.error(f'负类[{topic}] 批降到 {batch} 仍失败, 该类只拿到 {len(out)}/{count}')
            break
        batch //= 2
        logger.warning(f'负类[{topic}] 批大小降为 {batch} 重试')
    return out[:count]


def gen_negatives(client, rebuild=False):
    """函数功能: 全部主题 -> 负样本列表(逐主题缓存)。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = []
    for topic, count, desc in NEGATIVE_TOPICS:
        cache_file = os.path.join(CACHE_DIR, f'neg_{topic}.json')
        if not rebuild and os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            src = '缓存'
        else:
            questions = gen_negative_topic(client, topic, count, desc)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False)
            src = '新生成'
        print(f'  负类[{topic}] {len(questions)} 条 ({src})', flush=True)
        # source 字段只用于事后抽查各类占比, 训练不读它
        rows.extend({'query': q, 'label': '通用知识', 'source': topic} for q in questions)
    return rows


# ============================================================
# 合并 / 去重 / 配平 / 落盘
# ============================================================
def _norm(q):
    """去重键: 去掉所有标点与空白后的文本。'刹车异响?' 与 '刹车异响！' 算同一条。"""
    return re.sub(r'[\s,，。.?!？!、;；:：\'"“”‘’()（）]', '', q)


def build(pdf_path=DEFAULT_PDF, rebuild=False):
    """函数功能: 生成 -> 去重 -> 两类配平到 TARGET_PER_CLASS -> 写 JSON Lines。返回 dict 统计。"""
    client = get_llm_client()

    pos = gen_positives(client, pdf_path, rebuild=rebuild)
    neg = gen_negatives(client, rebuild=rebuild)

    # 去重。**必须跨类去重**: 同一条问题同时占两类, 训练时是纯噪声(同一输入两个标签),
    #   这种撞车在"提车但答不了"硬负样本上真会出现。
    seen = set()
    pos_uniq, neg_uniq = [], []
    for rows, bucket in ((pos, pos_uniq), (neg, neg_uniq)):
        for r in rows:
            key = _norm(r['query'])
            if key in seen:
                continue
            seen.add(key)
            bucket.append(r)

    # 配平: 每类**随机**抽 n 条。用 rng.sample 而非前 n 条 —— pos_uniq[:n] 按页号序截断
    #   会专砍页号最大的一批(尾部技术资料/术语表一条不剩); 固定种子保证同输入同输出。
    n_pos_raw, n_neg_raw = len(pos_uniq), len(neg_uniq)
    rng = random.Random(42)
    n = min(n_pos_raw, n_neg_raw, TARGET_PER_CLASS)
    pos_uniq = rng.sample(pos_uniq, n)
    neg_uniq = rng.sample(neg_uniq, n)

    # 交错写入: 文件前一半全正类的话, 肉眼抽查时看不到负类。
    rows = []
    for i in range(n):
        rows.append(pos_uniq[i])
        rows.append(neg_uniq[i])

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 统计口径: 去重后、配平**前**的真实条数都要打出来, 否则"某一类被砍多少"会被藏掉。
    stats = {
        '正类原始': len(pos), '正类去重后': n_pos_raw,
        '负类原始': len(neg), '负类去重后': n_neg_raw,
        '配平后每类': n, '总计': len(rows), '输出': OUT_PATH,
    }
    if n < min(n_pos_raw, n_neg_raw):
        logger.warning(f'配平截断: 正类 {n_pos_raw} -> {n}, 负类 {n_neg_raw} -> {n}'
                       f'(不足的那一类限制了总量, 想补要加语料)')
    return stats


if __name__ == '__main__':
    rebuild = '--rebuild' in sys.argv

    # ① 语料必须在(PDF 被 .gitignore 挡, 换机器重下); 缺了给清晰报错, 而非几百次调用后造空气
    print(f'① 手册 PDF: {DEFAULT_PDF}')
    assert os.path.abspath(DEFAULT_PDF) == DEFAULT_PDF, 'PDF 路径必须是绝对路径'
    if not os.path.exists(DEFAULT_PDF):
        raise FileNotFoundError(f'手册 PDF 不存在: {DEFAULT_PDF}')

    # ② 先确认 PDF 抽得出字、可用页够撑目标量(不出网, 秒回), 否则后面几百次调用全白烧
    usable = list(iter_usable_pages(DEFAULT_PDF))
    need = TARGET_PER_CLASS / Q_PER_PAGE
    print(f'② 可用页 {len(usable)} 页(过滤 <{MIN_PAGE_CHARS} 字与目录页); '
          f'每页 {Q_PER_PAGE} 题 -> 最多 {len(usable) * Q_PER_PAGE} 条, '
          f'需要 >= {need:.0f} 页')
    assert len(usable) >= need, f'可用页不足: {len(usable)} < {need:.0f}'

    # ③ LLM 必须真通: chat_text 失败不抛异常、只返回 '错误: ' 前缀, 不先探会安静地全跑成空
    probe = get_llm_client().chat_text('只回复两个字: 可用')
    print(f'③ LLM 探针: {probe!r}')
    assert not probe.startswith(ERROR_PREFIX), f'LLM 不可用: {probe}'

    # ④ 真跑(耗时十几分钟; 断掉重跑会跳过已完成部分)
    stats = build(rebuild=rebuild)
    for k, v in stats.items():
        print(f'   {k}: {v}')

    # ⑤ 产物复检: 条数、两类严格均衡、字段完整
    with open(OUT_PATH, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    labels = [r['label'] for r in rows]
    n_pos = labels.count('专业咨询')
    print(f'⑤ 产物: {len(rows)} 条, 专业咨询 {n_pos}, 通用知识 {len(rows) - n_pos}')
    assert n_pos == len(rows) - n_pos, '两类必须严格均衡'
    assert all({'query', 'label'} <= set(r) for r in rows), '缺 query/label 字段'

    print('\nL3 训练数据生成 gen_intent_dataset 结束')