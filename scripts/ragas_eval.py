import json
import os
from datetime import datetime
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# ==================== 从 .env 读配置(密钥不进版本库) ====================
# 复用 base/config.py 的项目根 .env 加载约定(load_dotenv), 键名对齐 .env:
#   DASHSCOPE_BASE_URL / DASHSCOPE_MODEL / DASHSCOPE_API_KEY
# DashScope 走 OpenAI 兼容接口(/compatible-mode/v1), 直接喂 OpenAI 客户端。
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.config import Config

_conf = Config()
DASHSCOPE_API_KEY = _conf.DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL = _conf.DASHSCOPE_BASE_URL
LLM_MODEL = _conf.LLM_MODEL
if "sk-xxxx" in DASHSCOPE_API_KEY or "删除密钥" in DASHSCOPE_API_KEY:
    raise SystemExit("请先在 .env 配置 DASHSCOPE_API_KEY, 密钥不落在代码/版本库里")

# RAGAS 裁判 LLM: langchain ChatOpenAI + LangchainLLMWrapper(与 eval_rag.py judge_llm 一致)。
# ⚠️ 不能走 ragas 的 llm_factory(client=...) —— ragas 0.2.6 该工厂不接受 client 关键字(实测报错);
#    且指标走异步路径 agenerate_text, ChatOpenAI 重写了 _agenerate, 不重写 _generate 也无须 temperature
#    适配器(DashScope 接受调用期 temperature, 见 eval_rag.py judge_llm 的论证)。
from langchain_openai import ChatOpenAI

llm = LangchainLLMWrapper(ChatOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    model=LLM_MODEL
))

# ==================== 配置Embedding模型 ====================
# 评估 embedding 用本地 BGE-M3(免费、中文好), 与 eval_rag.py 同一套加载方式。
# 用 milvus_model.hybrid 从本地 rag_qa/models/bge-m3/ 加载、钉死 CPU:
#  ⚠️ device 与 devices 都得设 'cpu' —— BGE-M3 默认会偷偷跑上 GPU(fp32),
#     4GB 卡上 ColQwen2 已占大半显存, 只在 devices(真正管目标设备)上设会 OOM。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _local_embeddings():
    from milvus_model.hybrid import BGEM3EmbeddingFunction

    fn = BGEM3EmbeddingFunction(
        model_name_or_path=os.path.join(_ROOT, 'rag_qa', 'models', 'bge-m3'),
        use_fp16=False, device='cpu', devices='cpu')

    class _BgeM3:
        def embed_documents(self, texts):
            return [list(v) for v in fn(list(texts))['dense']]

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    return LangchainEmbeddingsWrapper(_BgeM3())

# _local_embeddings() 返回的已是 LangchainEmbeddingsWrapper(_BgeM3()), 直接用, 别再包一层。
embeddings = _local_embeddings()


# ==================== 数据加载模块 ====================
def load_evaluation_data(json_file_path):
    """
    加载RAGAS评估数据集

    数据格式要求：
    - question: 用户问题
    - answer: RAG系统生成的回答
    - contexts: 检索到的上下文列表（字符串数组）
    - ground_truth: 标准答案（用于对比）
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ 成功加载评估数据 {len(data)} 条样本")
    for i, item in enumerate(data, 1):
        print(f"  {i}. {item['question'][:50]}...")

    return data


# ==================== 核心评估模块 ====================
def run_evaluation(json_file_path):
    """执行完整的RAGAS评估流程"""

    print("🚀 开始执行RAGAS评估...")
    print(f"📂 评估数据集路径: {json_file_path}")
    print(f"🤖 使用LLM模型: {LLM_MODEL} (via DashScope)")
    print("🔢 使用Embedding模型: 本地 BGE-M3 (rag_qa/models/bge-m3)")
    print("-" * 60)

    # 1. 加载数据
    eval_data = load_evaluation_data(json_file_path)

    # 转换为Hugging Face Dataset格式（RAGAS要求）
    dataset = Dataset.from_list(eval_data)

    # 2. 配置评估指标
    metrics = [
        Faithfulness(),  # 忠实度：答案是否基于上下文
        AnswerRelevancy(),  # 回答相关性：答案与问题的匹配度
        ContextPrecision(),  # 上下文精确率：检索内容的相关性
        ContextRecall(),  # 上下文召回率：是否检索到足够信息
        AnswerCorrectness()  # 回答正确性：与标准答案的对比
    ]

    # 3. 配置运行参数(与 eval_rag.py 一致的算术, 防止 ragas 默认重试睡眠拖死评估):
    #    RunConfig 默认 max_retries=10 配 wait_random_exponential(max=60), 9 次重试的纯等待上限
    #    是 1+2+4+8+16+32+60+60+60=243s, 光睡就把预算睡穿; 改成最多**尝试 3 次**(=首次+2 重试)。
    #    ⚠️ ragas 把 max_retries 喂给 tenacity 的 stop_after_attempt, 那是"尝试次数"不是"重试次数"。
    run_config = RunConfig(
        max_workers=1,
        max_retries=3,
        max_wait=5,
        timeout=60
    )

    # 4. 执行评估
    print("\n⏳ 正在执行评估计算...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        show_progress=True,  # 显示进度条
        raise_exceptions=False  # 遇到错误继续执行
    )

    # 5. 结果处理与可视化
    df = results.to_pandas()

    # 合并原始数据便于查看
    original_df = pd.DataFrame(eval_data)
    df = pd.concat([original_df, df], axis=1)

    # 6. 打印评估报告
    print_evaluation_report(df)

    # 7. 保存结果
    save_results(df)

    return results


def print_evaluation_report(df):
    """打印格式化的评估报告"""

    metric_keys = {
        'faithfulness': '忠实度',
        'answer_relevancy': '回答相关性',
        'context_precision': '上下文精确率',
        'context_recall': '上下文召回率',
        'answer_correctness': '回答正确性'
    }

    # 汇总表头
    print("\n" + "=" * 80)
    print("📊 评估结果汇总")
    print("=" * 80)

    print(f"\n{'问题':<60} {'忠实度':<10} {'相关性':<10} {'精确率':<10} {'召回率':<10} {'正确性':<10}")
    print("-" * 110)

    # 逐条显示
    for idx, row in df.iterrows():
        question = row['question'][:57] + '...' if len(row['question']) > 57 else row['question']

        # 安全获取数值（处理可能的None值）
        def safe_format(value):
            return f"{value:.4f}" if isinstance(value, (int, float)) else 'N/A'

        print(f"{question:<60} "
              f"{safe_format(row.get('faithfulness')):<10} "
              f"{safe_format(row.get('answer_relevancy')):<10} "
              f"{safe_format(row.get('context_precision')):<10} "
              f"{safe_format(row.get('context_recall')):<10} "
              f"{safe_format(row.get('answer_correctness')):<10}")

    # 平均分
    print("\n" + "=" * 80)
    print("📈 平均指标得分")
    print("=" * 80)

    for key, label in metric_keys.items():
        if key in df.columns:
            avg_value = df[key].mean()
            print(f"  {label} ({key}): {avg_value:.4f}")
        else:
            print(f"  {label} ({key}): N/A")

    # 详细结果
    print("\n" + "=" * 80)
    print("📝 详细评估结果")
    print("=" * 80)

    for idx, row in df.iterrows():
        print(f"\n样本 {idx + 1}:")
        print(f"❓ 问题: {row['question']}")
        print(f"💬 回答: {str(row['answer'])[:100]}...")
        print(f"📚 上下文数量: {len(row['contexts'])}")
        print(f"✅ 标准答案: {str(row['ground_truth'])[:100]}...")
        print("\n📊 指标得分:")

        for key, label in metric_keys.items():
            if key in df.columns:
                value = row[key]
                if isinstance(value, (int, float)):
                    print(f"  • {label}: {value:.4f}")
                else:
                    print(f"  • {label}: {value}")
        print("-" * 60)


def save_results(df):
    """保存评估结果到文件"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(__file__)

    # 保存为JSON
    json_path = os.path.join(base_dir, f"ragas_eval_results_{timestamp}.json")
    df.to_json(json_path, orient='records', force_ascii=False, indent=2)
    print(f"\n✅ 评估结果已保存至JSON: {json_path}")

    # 保存为Excel（多工作表）
    excel_path = os.path.join(base_dir, f"ragas_eval_results_{timestamp}.xlsx")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 工作表1：详细结果
        df.to_excel(writer, sheet_name='详细评估结果', index=False)

        # 工作表2：平均得分
        metric_keys = {
            'faithfulness': '忠实度',
            'answer_relevancy': '回答相关性',
            'context_precision': '上下文精确率',
            'context_recall': '上下文召回率',
            'answer_correctness': '回答正确性'
        }

        avg_scores = {}
        for key, label in metric_keys.items():
            if key in df.columns:
                avg_scores[label] = df[key].mean()

        avg_df = pd.DataFrame({
            '指标': list(avg_scores.keys()),
            '平均得分': list(avg_scores.values())
        })
        avg_df.to_excel(writer, sheet_name='平均指标得分', index=False)

    print(f"✅ 评估结果已保存至Excel: {excel_path}")
    print(f"   • 工作表1: 详细评估结果")
    print(f"   • 工作表2: 平均指标得分")


if __name__ == "__main__":
    # 默认加载同目录下的数据集
    json_file_path = os.path.join(os.path.dirname(__file__), "ragas_eval_dataset.json")
    run_evaluation(json_file_path)
