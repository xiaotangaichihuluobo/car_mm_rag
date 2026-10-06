# 本地 ColQwen2-2B: 页图 / 查询文字 -> patch 多向量(同一空间)。
# ColPali 系模型用同一骨干编码文档页图(process_images)与查询文字(process_queries),
#   产出都在 (seq,128) 空间 —— 这是"文字查询对页图 muvera+maxsim"的前提。
# 硬规则 R1: 只走本地权重禁 API(建库与查询必须同权重, 否则向量空间对不上=静默失效)。

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from base.config import Config
from base.logger import logger

conf = Config()

# 模块级缓存: 模型只加载一次, 4GB 卡上加载约 11s
_ENCODER_CACHE = None


class ColQwen2Encoder(object):
    """用本地 ColQwen2-2B(4bit nf4)把页图/用户图编码成 patch 多向量。只处理图片。"""

    def __init__(self, model_path=None, device=None):
        # 权重目录优先用 config 指定, 勿重下
        self.model_path = model_path or conf.MM_MODEL_PATH
        self.modelscope_id = conf.MM_MODELSCOPE_ID

        # 模型延迟到首次编码才加载, 只建库不查询的场景不必占显存
        self.model = None
        self.processor = None

        # torch 在这里 import, 只有编码路径用得到
        if device is not None:
            self.device = device
        else:
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                self.device = 'cpu'

    def _resolve_source(self):
        """权重来源: 本地目录 -> ModelScope(公开仓, 无需 API key)。"""
        if self.model_path and os.path.isdir(self.model_path):
            has_weight = any(f.endswith(('.safetensors', '.bin'))
                             for f in os.listdir(self.model_path))
            if has_weight:
                return self.model_path
            logger.warning(f"本地模型目录无权重文件, 将继续查找: {self.model_path}")

        logger.warning(f"本地权重缺失, 尝试从 ModelScope 拉取 {self.modelscope_id}")
        try:
            from modelscope import snapshot_download
            return snapshot_download(self.modelscope_id)
        except Exception as e:
            raise RuntimeError(
                f"ColQwen2 权重不可用: 本地 {self.model_path} 无权重, ModelScope 拉取也失败({e})。"
                f"请确认 {self.model_path} 下有 model.safetensors。"
            )

    def _ensure_loaded(self):
        """加载模型(4bit nf4; bf16 在 4GB 卡放不下, int8 约 3.5GB 太紧)。幂等。"""
        if self.model is not None:
            return

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, ColQwen2ForRetrieval

        source = self._resolve_source()
        logger.info(f"加载 ColQwen2: {source} (device={self.device})")

        quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_quant_type='nf4',
                                          bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = ColQwen2ForRetrieval.from_pretrained(
            source,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map='auto')
        self.processor = AutoProcessor.from_pretrained(source)
        self.model.eval()
        logger.info("ColQwen2 已就绪")

    def encode_images(self, images, show_progress=False, progress_every=25):
        """
        页图/用户图 -> patch 多向量 (n,128)。

        :param images: PIL.Image 列表
        :param show_progress: 建库时开(354 页要跑几分钟, 不给进度会以为卡死)
        :param progress_every: 每编码多少张报一次进度
        :return: [numpy(n_i, 128), ...]
        """
        import torch

        self._ensure_loaded()

        out = []
        total = len(images)
        with torch.inference_mode():
            for i, img in enumerate(images):
                batch = self.processor.process_images(images=[img])

                inputs = {}
                for key, value in batch.items():
                    if hasattr(value, 'to'):    # 只把张量搬上设备
                        inputs[key] = value.to(self.device)

                embeddings = self.model(**inputs).embeddings[0]  # (seq, 128) bf16

                if 'attention_mask' in inputs:
                    mask = inputs['attention_mask'][0].bool()
                else:
                    mask = torch.ones(embeddings.shape[0],
                                      dtype=torch.bool,
                                      device=self.device)

                # 去掉 padding 位; bf16 先转 float32 才能落 numpy
                out.append(embeddings[mask].float().detach().cpu().numpy())

                del batch, inputs, embeddings
                torch.cuda.empty_cache()  # 4GB 卡, 不主动回收会碎

                if show_progress and ((i + 1) % progress_every == 0 or i + 1 == total):
                    logger.info(f"  编码页图 {i + 1}/{total}")

        return out

    def encode_text(self, texts, show_progress=False, progress_every=20):
        """
        查询文字 -> patch 多向量 (n,128)。与 encode_images **同一空间**。

        ColPali 的查询侧: processor.process_queries 把文字拼成文字 token 序列,
        与页图 patch 在同一骨干输出 (seq,128) —— 检索就是拿 query patch 对页 patch
        做 MaxSim。建库与查询必须同权重(R1), 否则空间对不上=静默失效。

        :param texts: 查询文字 list
        :param show_progress: 批量时开
        :return: [numpy(n_i, 128), ...]
        """
        import torch

        self._ensure_loaded()

        out = []
        total = len(texts)
        with torch.inference_mode():
            for i, text in enumerate(texts):
                batch = self.processor.process_queries(text)

                inputs = {}
                for key, value in batch.items():
                    if hasattr(value, 'to'):    # 只把张量搬上设备
                        inputs[key] = value.to(self.device)

                embeddings = self.model(**inputs).embeddings[0]  # (seq, 128) bf16

                if 'attention_mask' in inputs:
                    mask = inputs['attention_mask'][0].bool()
                else:
                    mask = torch.ones(embeddings.shape[0],
                                      dtype=torch.bool,
                                      device=self.device)

                # 去 padding; bf16 先转 float32 才能落 numpy
                out.append(embeddings[mask].float().detach().cpu().numpy())

                del batch, inputs, embeddings
                torch.cuda.empty_cache()  # 4GB 卡, 不主动回收会碎

                if show_progress and ((i + 1) % progress_every == 0 or i + 1 == total):
                    logger.info(f"  编码文字 {i + 1}/{total}")

        return out


def get_encoder():
    """模块级单例: 整个进程共用一份 ColQwen2, 显存只占一次。"""
    global _ENCODER_CACHE
    if _ENCODER_CACHE is None:
        _ENCODER_CACHE = ColQwen2Encoder()
    return _ENCODER_CACHE