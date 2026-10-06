
# L3 意图识别: 加载 BERT 前向 -> 判「通用知识」直答 / 「专业咨询」检索。
# 训练不在这里(离线在 train_classifier.py, 权重写到 bert_query_classifier, 本文件只加载)

import os
import sys

# ---- 路径引导: 把项目根放进 sys.path, 让本文件既能被 import, 也能直接 python 运行 ----
# 本文件在 rag_qa/core/, 往上退 2 层到项目根。(家规 A8: 只插项目根)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# rag_qa/ 目录本身 —— 本模块的模型权重与训练数据都挂在它下面。
#   train_classifier.py 也 import 这一条, 保持"rag_qa 在哪"只有一个来源。
RAG_QA_DIR = os.path.dirname(_HERE)

import torch

from base.logger import logger
from transformers import BertForSequenceClassification, BertTokenizer


class QueryClassifier:
    """职责: 把用户问题分成「通用知识」/「专业咨询」两类。"""

    def __init__(self, model_path=None):
        # 模型路径必须由 __file__ 推导为项目内绝对路径: 相对路径换个目录启动就加载不到
        #   权重, 会**静默退回随机初始化模型**, 结果无意义。别改回相对路径。
        self.model_path = model_path if model_path else os.path.join(
            RAG_QA_DIR, 'models', 'bert_query_classifier')
        self.bert_path = os.path.join(RAG_QA_DIR, 'models', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.model = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'使用设备: {self.device}')

        # 标签映射: 通用知识 -> 0, 专业咨询 -> 1。必须与训练时的 label_map 一致
        self.label_map = {"通用知识": 0, "专业咨询": 1}

        self.load_model()

    def load_model(self):
        """
        函数功能: 优先从 self.model_path 加载已训练权重, 不存在则用 bert-base-chinese 初始化一个。

        【已知行为: 模型文件不存在时**静默**退回随机初始化的新模型】
        此时分类结果无意义, 但**没有任何标记**能区分这两种情况 —— 编排器只用
        predict_proba 的 label 与 confidence, 没有可读的训练状态。真的缺权重时,
        唯一的现象是分类结果看起来乱。排查前先确认这个目录在不在:
        rag_qa/models/bert_query_classifier。
        """
        if os.path.exists(self.model_path):
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info(f"加载模型: {self.model_path}")
        else:
            logger.warning(f"未找到已训练权重 {self.model_path}, 退回未训练的 bert-base-chinese")
            self.model = BertForSequenceClassification.from_pretrained(self.bert_path, num_labels=2)
            self.model.to(self.device)

    def _infer(self, query):
        """
        单条查询前向推理 -> 两类概率。
        :return: (label:str, confidence:float, probs:list[float])
        """
        if self.model is None:
            logger.error('模型未加载, 无法分类')
            return '通用知识', 0.0, []

        # max_length 与训练时一致(128), 不一致会改变分布
        encoding = self.tokenizer(
            query, truncation=True, padding=True, max_length=128, return_tensors='pt')
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self.model(**encoding).logits

        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        idx = int(torch.argmax(logits, dim=1).item())
        return ('专业咨询' if idx == 1 else '通用知识'), probs[idx], probs

    def predict_proba(self, query):
        """
        函数功能: L3 意图识别 —— 判「通用知识」还是「专业咨询」, 并带出置信度与各类概率。

        只暴露这一个方法: 要标签直接取返回值的第一个元素。

        :return: (label, confidence, probs)
        """
        return self._infer(query)


if __name__ == '__main__':
    # ============================================================
    # 测试代码(仅直接运行本文件时执行)
    #   跑法: python rag_qa/core/query_classifier.py
    #   需要 rag_qa/models/bert-base-chinese(tokenizer); 没有已训练权重时会明确告警。
    # ============================================================
    clf = QueryClassifier()

    # ① 模型路径必须是**项目内绝对路径**。
    #   这条是专门钉住那个坑的: 改成相对路径后, 从别的目录启动就会静默加载失败,
    #   而现象只是"分类结果看起来乱" —— 不报错, 所以只能靠断言挡。
    expect = os.path.join(RAG_QA_DIR, 'models', 'bert_query_classifier')
    print(f'① 模型路径: {clf.model_path}')
    assert clf.model_path == expect, f'模型路径应为 {expect}, 实得 {clf.model_path}'
    assert os.path.isabs(clf.model_path), '模型路径必须是绝对路径'

    # ② 权重目录在不在 —— 不在就意味着下面 ③ 的结果无意义(见 load_model 的说明)
    if not os.path.exists(clf.model_path):
        print(f'② [警告] {clf.model_path} 不存在, 分类结果无意义; '
              f'先跑 python rag_qa/core/train_classifier.py')

    # ③ 推理接口的形状: 标签合法、两类概率、和为 1
    label, confidence, probs = clf.predict_proba('电动尾门怎么打开?')
    print(f'③ predict_proba -> label={label} confidence={confidence:.4f} probs={probs}')
    assert label in clf.label_map, f'标签非法: {label}'
    assert len(probs) == 2, f'应为二分类, 实得 {len(probs)} 类'
    assert abs(sum(probs) - 1.0) < 1e-4, f'两类概率之和应为 1, 实得 {sum(probs)}'
    assert abs(probs[clf.label_map[label]] - confidence) < 1e-6, \
        'confidence 应等于所判类别那一项概率'

    # ④ 边界断言: 两个方向各钉两条。
    #   【为什么必须有这条】形状类断言(标签合法、概率和为 1)全绿, 不代表模型有用:
    #   实测模型可以把**每一个**汽车问题都判成「通用知识」(8/8, 置信度 0.947~0.970) ——
    #   判「通用知识」= qwen-plus 直答, 于是 L4 的手册检索在默认路径下**一次都走不到**。
    #   这四条钉的是**边界**, 模型或训练数据一换就该重跑。
    #   选例原则: 取语义上毫无歧义的 —— 正类用手册明写的内容, 负类用与汽车手册
    #   完全无关的主题。**别把「今天天气怎么样?」这类短而含糊的句子选进来**:
    #   实测它会判成「专业咨询」(0.998), 而那是当前模型已知的偏斜方向(见 CLAUDE-full.md
    #   环境事实表「L3 短含糊问句偏斜」), 拿它当断言会变成一个必挂的测试。
    for bad_or_good, expect in [
        ('发动机异响是什么原因?', '专业咨询'),
        ('刹车片多久换一次?', '专业咨询'),
        ('Java零基础学费多少?', '通用知识'),
        ('考研数学怎么复习?', '通用知识'),
    ]:
        got, conf, _ = clf.predict_proba(bad_or_good)
        print(f'④ {got} {conf:.4f}  {bad_or_good}   (期望 {expect})')
        assert got == expect, f'「{bad_or_good}」应判 {expect}, 实得 {got}({conf:.4f})'

    # (原 ④「predict 与 predict_proba 标签必须一致」随 predict() 一并删除 ——
    #  被验的方法已经不在了, 留着是空断言。)

    print('\nL3 意图识别 query_classifier 自检 PASS')
