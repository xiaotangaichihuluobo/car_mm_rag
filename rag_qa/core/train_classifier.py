# L3 意图分类器的离线训练(一次性): 读 json -> 分词 -> Trainer 3 轮 -> 存权重 -> 报告。
# 跑法: PYTHONPATH=. python rag_qa/core/train_classifier.py (权重写到 bert_query_classifier, 供 query_classifier.py 加载)
# 【闸】权重目录还在直接 RuntimeError 拒训(在见过全部页的模型上算的数都是假的); 要重头训先移走目录

import hashlib
import json
import os
import sys
import time

# ---- 路径引导: 把项目根放进 sys.path, 让本文件既能被 import, 也能直接 python 运行 ----
# 本文件在 rag_qa/core/, 往上退 2 层到项目根。(家规 A8: 只插项目根)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from transformers import Trainer, TrainingArguments
import torch

from base.logger import logger
from rag_qa.core.query_classifier import RAG_QA_DIR, QueryClassifier

# 训练集 json 的缺省位置: 项目内绝对路径, 不受运行目录影响。
# 命令用汽车域数据 car_intent_5000.json(旧 IT 培训域的 model_generic_5000.json 485 条已不用):
# 旧数据教出的边界是「IT培训 -> 专业咨询」, 在汽车项目里**正好是反的** —— 实测汽车问题
# 8/8 被判成「通用知识」直答, 于是 L4 手册检索在默认路径下永远走不到。
# 新数据由 rag_qa/classify_data/gen_intent_dataset.py 从 car_mm 实际入库的那份手册
# (领克, 354 页)逐页生成, 正类 2500 / 负类 2500。
DEFAULT_DATA_FILE = os.path.join(RAG_QA_DIR, 'classify_data', 'car_intent_5000.json')

# 中间产物(checkpoint / 日志)的落点。同样锚在 rag_qa/ 下 —— 见 train_model 里的说明。
TRAIN_OUTPUT_DIR = os.path.join(RAG_QA_DIR, 'bert_results')


class ClassifierTrainer(QueryClassifier):
    """
    职责: 在 QueryClassifier(加载 + 推理)之上, 加训练、评估、落盘。
    用继承复用 tokenizer / label_map / model_path, 两者依赖的库完全不同。
    """

    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"保存模型至: {self.model_path}")

    def preprocess_data(self, texts, labels):
        """
        函数功能: 对文本进行分词, 阶段, 填充, 将标签转换为: 数字.
        :param texts: 待处理的文本列表.
        :param labels: 文本对应的标签列表.
        :return: 处理后的编码 (input_ids, attention_mask) 和 数字标签列表.
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings  # (input_ids, attention_mask)
                self.labels = labels        # 数字标签列表

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def train_model(self, data_file=None, group_by_page=True):
        """
        训练BERT分类模型, 区分查询分类为'通用知识' 和 '专业咨询'
        :param data_file: 数据集文件路径; None 则用 rag_qa/classify_data/car_intent_5000.json
        :param group_by_page: 划分方式。True(主指标)=按 page 分组划, 消除同页泄漏;
            False=旧的随机划分, 只为出对照数, 不作为主指标。
        :return: 无
        """
        # 0. 拒在已存在权重上开训(见下面那道闸)。
        #
        # 【为什么闸设在 train_model 而非 __main__】__init__ 已调 load_model(): 权重在即
        #   旧权重接着训, 而 train_model 会把评估数落盘到 intent_result.json 的 by_page 键、
        #   标成 eval_intent 的「主指标」—— 在一个见过全部页的模型上调出的数会变成谎话。
        #   真实跑法是命令行 `ClassifierTrainer().train_model(...)`, 闸在 __main__ 挡不住。
        #   重训: 把权重目录 mv 走(备份勿删)再跑(见文件头)。
        if os.path.exists(self.model_path):
            raise RuntimeError(
                f'已存在已训练权重 {self.model_path}, 拒绝在其上接着训: 这次算出来的准确率'
                f'会落到 intent_result.json 的 by_page 键上, 那个数将是假的(模型见过全部页)。'
                f'确需重训, 先按本文件头注释的做法把它移走再跑, 例如: '
                f'mv "{self.model_path}" "{self.model_path}.bak.<时间戳>"'
            )

        # 0.1 缺省数据文件: 由 __file__ 推导的绝对路径(旧实现死写相对 CWD, 换目录即 FNF)。
        if data_file is None:
            data_file = DEFAULT_DATA_FILE

        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]

        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        # 划分训练/验证集 80%/20%, 固定随机种子可复现。
        #
        # 【按页分组非随机】专业咨询 2500 条逐页生成, 同页题用词重叠; 随机划分让同页题同入
        #   训练和验证, 模型背"这页的味道"而非真学会, 准确率虚高。按 page 分组=整页同侧,
        #   才是"没见过的页判不判得对"。通用知识 2500 条无 page 字段(通用语料), 各自成组。
        #   group_by_page=False 保留旧行为只为出对照数, 非主指标。
        if group_by_page:
            groups = [item.get('page', f'g{i}') for i, item in enumerate(data)]
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(splitter.split(texts, labels, groups=groups))
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            # 组数一并打: 缺 page 部分各自成组; 若无 page 则按页划分退化逐条随机, 却仍以
            # by_page 落盘。打组数可辨: 本数据 292 真页 + 2500 单例 = 2792 组 vs 退化 5000 组。
            logger.info(f'按页分组划分: 训练 {len(train_texts)} / 验证 {len(val_texts)} / '
                        f'组数 {len(set(groups))}(其中带 page 的 '
                        f'{len({item["page"] for item in data if "page" in item})} 组)')
        else:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            logger.info(f'随机划分: 训练 {len(train_texts)} / 验证 {len(val_texts)}')

        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        # 配置训练参数。output_dir 锚到 rag_qa/ 下(原相对 CWD 会在运行目录堆 G 级 checkpoint,
        #   历史上在 rag_qa/core/ 堆过一个 1.2G 的 bert_results)。
        training_args = TrainingArguments(
            output_dir=TRAIN_OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=20,
            weight_decay=0.01,
            logging_dir=os.path.join(TRAIN_OUTPUT_DIR, 'logs'),
            logging_steps=10,
            # 参数名须用 eval_strategy: transformers ≥4.46 移除 evaluation_strategy, 4.57.6 传旧名直接
            #   TypeError("unexpected keyword argument 'evaluation_strategy'")。别改回去。
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            fp16=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        logger.info("开始训练 BERT 模型...")
        trainer.train()
        self.save_model()

        # 评估。val_labels 已被 preprocess_data 重绑定成数字标签(len 仍是验证集条数)。
        report = self.evaluate_model(val_texts, val_labels)
        self.save_intent_result(report, len(val_labels), group_by_page, data_file)

    def compute_metrics(self, eval_pred):
        """
        计算分类任务的评估指标(准确率)
        :param eval_pred:  包含模型输出logits 和 真实标签的元素.
        :return: 准确率字典.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def evaluate_model(self, texts, labels):
        """
        在给定文本和标签上评估模型, 输出分类报告和混淆矩阵.
        :param texts:  待评估的文本列表
        :param labels: 文本对应的真实标签(数字形式)
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        dataset = self.create_dataset(encodings, labels)

        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)

        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = labels  # 直接使用数字标签

        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "专业咨询"]   # 指定标签名称, 使报告更易读.
        ))

        # 混淆矩阵: 展示预测标签 和 真实标签的匹配情况.
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

        # 再算一份 dict 形式的报告: 字符串没法取数, 落盘要用 dict(对验证集重算一遍很便宜)。
        return classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "专业咨询"],
            output_dict=True,
        )

    def save_intent_result(self, report, n_val, group_by_page, data_file):
        """
        把 L3 的评估数字落一份 JSON, 给 scripts/eval_rag.py 的 eval_intent() 读。

        两个口径分别存而非覆盖: 随机划分虚高(同页泄漏), 按页才是真指标。

        每口径都带 provenance: 否则新旧数长得一样, 无法判断还作不作数。记训练数据 sha256
        与权重目录 mtime, 两者可就地复核; 只写事实, 写不准的东西宁可不写, 错的指纹比没有更坏。
        (落盘在 save_model() 之后, 故 model_mtime 即本次权重。)
        """
        out_dir = os.path.join(RAG_QA_DIR, 'data', 'eval')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'intent_result.json')

        previous = {}
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                previous = json.load(f)

        # 数据指纹: 对训练 json 本体算 sha256。文件不到 1MB, 便宜且可复核。
        with open(data_file, 'rb') as f:
            data_sha256 = hashlib.sha256(f.read()).hexdigest()

        key = 'by_page' if group_by_page else 'random'
        previous[key] = {
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            '通用知识_f1': report['通用知识']['f1-score'],
            '专业咨询_f1': report['专业咨询']['f1-score'],
            'n_val': n_val,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'provenance': {
                'data_file': data_file,
                'data_file_sha256': data_sha256,
                'model_path': self.model_path,
                'model_mtime': time.strftime(
                    '%Y-%m-%d %H:%M:%S',
                    time.localtime(os.path.getmtime(self.model_path))),
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(previous, f, ensure_ascii=False, indent=2)
        logger.info(f"L3 评估结果已落盘: {path} (口径={key})")


if __name__ == '__main__':
    # 1. 数据集必须在 (缺省 = rag_qa/classify_data/car_intent_5000.json 的绝对路径)。
    #    这条断言是给"换个目录跑训练"挡路的: 路径改成相对 CWD 后, 第一个现象就是这里炸。
    print(f'① 训练数据: {DEFAULT_DATA_FILE}')
    assert os.path.isabs(DEFAULT_DATA_FILE), '训练数据路径必须是绝对路径'
    assert os.path.exists(DEFAULT_DATA_FILE), f'训练数据不存在: {DEFAULT_DATA_FILE}'
    with open(DEFAULT_DATA_FILE, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f.readlines()]
    print(f'   共 {len(rows)} 条, 字段: {sorted(rows[0].keys())}')
    assert {'query', 'label'} <= set(rows[0]), f'训练数据缺字段: {sorted(rows[0].keys())}'

    # 2. 标签必须都能被 label_map 覆盖 —— 缺一个就是 preprocess_data 里的 KeyError
    trainer_obj = ClassifierTrainer()
    unknown = sorted({r['label'] for r in rows} - set(trainer_obj.label_map))
    print(f'② 标签集合: {sorted({r["label"] for r in rows})}')
    assert not unknown, f'训练数据里有 label_map 覆盖不到的标签: {unknown}'

    # 3. 训练中间产物目录: 必须是项目内绝对路径(否则在哪个目录跑就在哪里堆 checkpoint)
    print(f'③ 训练产物目录: {TRAIN_OUTPUT_DIR}')
    assert os.path.isabs(TRAIN_OUTPUT_DIR), '训练产物目录必须是绝对路径'

    # 4. 真正开训(3 轮, 约数分钟; 存权重 + 出分类报告/混淆矩阵到日志)
    #    注意: 上面第 2 步的 ClassifierTrainer() 一旦加载到已存在的权重, 这里就会
    #    被 train_model 开头那道闸 RuntimeError 挡住 —— 那是故意的, 不是坏了。
    #    想真跑: 先把 rag_qa/models/bert_query_classifier 移走(见文件头「已知行为」那段)。
    trainer_obj.train_model()

    print('\nL3 分类器训练 train_classifier 结束')
