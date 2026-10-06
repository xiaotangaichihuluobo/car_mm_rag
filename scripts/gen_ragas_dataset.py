# -*- coding: utf-8 -*-
"""
gen_ragas_dataset.py —— 离线生成 ragas 评估数据集(JSON)。
从 eval_set.json 逐题现造 answer/contexts 落盘, 供 ragas_eval.py 只读评估用。

【为什么单独一个脚本】评估过程不应边检索边打分(慢、不可复现、还撞 ragas bug)。
  先把数据固定成 JSON, 打分脚本只读文件, 快且每跑一次分数可追溯。

【产物】rag_qa/data/eval/ragas_eval_dataset.json
  每行(数组元素): {id, question, answer, contexts, reference}
   - question        : eval_set 的域内题
   - contexts        : 检索 top5 候选页全文(list)
   - answer          : qwen-plus 对着候选页生成的回答(被评测对象)
   - reference       : qwen-plus 对着 gold 页生成的标准答案(评测基准, 不参与打 field)
"""

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in os.sys.path:
    os.sys.path.insert(0, _ROOT)

from base.config import Config
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

conf = Config()
EVAL_SET = os.path.join(_ROOT, 'rag_qa', 'data', 'eval', 'eval_set.json')
OUT = os.path.join(_ROOT, 'rag_qa', 'data', 'eval', 'ragas_eval_dataset.json')


def build_llm():
    return ChatOpenAI(
        model=conf.LLM_MODEL,
        api_key=conf.DASHSCOPE_API_KEY,
        base_url=conf.DASHSCOPE_BASE_URL,
        temperature=0.1,
    )


def page_text(pdf, page):
    return pdf[page].get_text('text')


def main(n=0):
    with open(EVAL_SET, encoding='utf-8') as f:
        rows = json.load(f)
    dom = [r for r in rows if r['id'].startswith('q') and r.get('gold_pages')]
    if n:
        dom = dom[:n]

    llm = build_llm()
    from rag_qa.core.image_leg import PageRetriever
    retriever = PageRetriever(load_on_init=True)
    import fitz
    pdf = fitz.open(conf.MM_PDF_PATH)

    dataset = []
    for r in dom:
        hits = retriever.image_leg(r['question'], k=5)
        contexts = [page_text(pdf, page) for _sid, page, _score in hits]
        contexts = [c for c in contexts if isinstance(c, str) and c.strip()]

        # 候选页 -> answer (被评测的回答)
        docs = '\n---\n'.join(contexts)
        prompt = ChatPromptTemplate.from_messages([
            ('system', '你是汽车使用手册问答助手。请只依据下面提供的资料回答, 资料没有的内容就说不知道。'),
            ('human', '资料：\n{docs}\n\n问题：{question}'),
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({'docs': docs, 'question': r['question']})

        # gold 页 -> reference (标准答案, 独立于检索, 只当评测基准)
        gold_texts = [page_text(pdf, p) for p in r['gold_pages']]
        gold_texts = [c for c in gold_texts if isinstance(c, str) and c.strip()]
        prompt_ref = ChatPromptTemplate.from_messages([
            ('system', '你是汽车使用手册问答的标准答案撰写者。请只依据下面资料, 用一句通顺的话准确回答问题。'),
            ('human', '资料：\n{gold}\n\n问题：{question}'),
        ])
        ref = (prompt_ref | llm | StrOutputParser()).invoke(
            {'gold': '\n---\n'.join(gold_texts), 'question': r['question']})

        dataset.append({
            'id': r['id'],
            'question': r['question'],
            'answer': answer,
            'contexts': contexts,
            'reference': ref,
        })
        print(f"[{r['id']}] {r['question']}")

    pdf.close()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f'\n已生成 {len(dataset)} 条 -> {OUT}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0, help='只生成前 n 条(0=全部)')
    args = ap.parse_args()
    main(args.n)