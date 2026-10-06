# 级联里唯一出网环节: prompt 模板 + 图片编解码 + qwen 两条调用(L3 直答 / L4 生成)。
# R1/R2 边界: 检索全本地, 只有拿候选页生成答案才允许出网
#
# 为什么不用流式:
#   L4 的 prompt 强制模型先输出「证据: 足够/不足」再由上层决定答还是拒(见 vl_prompt)。
#   若边生成边推给前端, 等看到"不足"时错误内容早就吐出去了, 自评这道闸门形同虚设。
#   所以这里一律整段取回; 前端要打字机效果, 由 app.py 把最终字符串切块推送即可。
# ============================================================

import base64
import io

from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from PIL import Image

from base.config import Config
from base.logger import logger


conf = Config()

# 调用失败时返回的哨兵前缀。`new_main.py` / `multimodal_qa.py` 以 startswith(ERROR_PREFIX) 判失败,
#   改动这个字符串要同步检查那些调用点。
ERROR_PREFIX = '错误: '

# 送模型前的图片长边上限。qwen-vl 按像素折 token, 手册页 1400px + 用户原图不缩的话,
#   5 张候选页轻松超十万 token —— 既慢又贵, 且超出上下文会被静默截断(表现为"模型没看见第 4、5 页")。
VL_IMAGE_MAX_SIDE = 1000


# ============================================================
# 一、图片编解码: PIL <-> base64 data URL, 以及送模型前的缩放
#   三处用到:
#     - 前端把用户图片以 base64 传上来 -> load_image_from_base64 还原成 PIL(app.py 调)
#     - 候选页图 / 用户原图送 qwen-vl 前要缩到可控像素 -> shrink_image
#     - 送出去时编成 data URL -> image_to_data_url
# ============================================================
def load_image_from_base64(data):
    """
    函数功能: 把前端传来的图片还原成 PIL.Image。
    :param data: 三种形态都收 ——
                 "data:image/png;base64,iVBOR..." (浏览器 FileReader 的默认可读格式)
                 "iVBOR..."                    (纯 base64)
                 bytes                          (已解码的字节)
    :return: PIL.Image(RGB)。解析失败返回 None, 让上层当作"没带图"。

    本函数不做缩放: 生产调用点都不传缩放参数, shrink_image 从未被走到; 缩放在前端(1400)做。
    """
    if not data:
        return None
    try:
        if isinstance(data, bytes):
            raw = data
        else:
            text = str(data)

            # 去掉 data URL 头, 只留逗号后面的 base64 正文。
            #   没有逗号(形如 'data:image/png;base64' 那种没正文的)就原样交给 b64decode 去报错
            if text.startswith('data:') and ',' in text:
                text = text.split(',', 1)[1]
            raw = base64.b64decode(text)
        img = Image.open(io.BytesIO(raw)).convert('RGB')
    except Exception:
        # 图片坏了/不是图片: 返回 None, 上层走"纯文本"分支。
        #   不让一张坏图把整个问答打挂 —— 带图问答降级成不带图, 比 500 好
        return None
    return img


def shrink_image(img, max_side):
    """
    函数功能: 按长边等比缩放图片(只缩不放)。
    :param img: PIL.Image
    :param max_side: 长边上限像素; None 或非正数表示不缩
    :return: 缩放后的 PIL.Image(原图本来就够小则原样返回)
    """
    if not max_side or max_side <= 0:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def image_to_data_url(img, quality=85):
    """
    函数功能: PIL.Image -> base64 data URL, **一律 JPEG**。
    送 qwen-vl 有两种写法: 传公网 URL(没有) 或传 base64 data URL(用这个)。
    :return: "data:image/jpeg;base64,...."

    恒 JPEG, 不收 `fmt` 形参: 全仓无一处传非 JPEG。
    """
    buf = io.BytesIO()
    # JPEG 不支持透明通道; 传进来的图已统一 convert('RGB'), 这里只做兜底
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buf, format='JPEG', quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


# ============================================================
# 二、prompt 模板: 两条链路真正在用的两个。
# ============================================================
def direct_prompt():
    """
    L3 直答模板: 意图识别判为「通用知识」时, 不检索, 直接用 qwen-plus 答。

    与检索类 prompt 的区别是没有 {context}: 这条路不检索, 空上下文会诱导模型编造。
    """
    return PromptTemplate(
        template="""
        你是一个专业、严谨的智能助手，正在回答用户的问题。

        要求：
        1. 只用你自己的知识回答，不要假装查阅了任何文档。
        2. 回答简洁准确，直接给结论，不要寒暄。
        3. 如果你不确定，就直说"这个问题我不太确定"，不要编造。

        对话历史: {history}
        问题: {question}

        回答:
        """,
        input_variables=["history", "question"],
    )


def vl_prompt():
    """
    L4 多模态生成模板: 看图作答, 并强制模型做「证据自评」。

    L4 的闸门3 靠模型自评: 检索分只说明"这页有点像", 不说明"真回答了问题"(BERT 意图也
    分不出"专业问题但手册里没有")。输出用"标签行"而非 JSON —— PromptTemplate 用花括号
    当占位符, 写 JSON 得全转义 {{ }}; 解析见 multimodal_qa.parse_vl_output。
    """
    return PromptTemplate(
        template="""
        你是汽车用户手册问答助手。用户会给你一个问题、可能的用户实拍照片，
        以及若干张从《用户手册》里检索出来的候选页图片。每张页图片按下列顺序标注了
        「来源(书号)|页码」的标识:
        候选页依次为: {pages_desc}

        请严格依据候选页图片的内容回答，并先做证据自评。按以下格式输出，不要有任何多余内容：

        证据: 足够 或 不足
        页码: 你实际依据的候选页标识(如 train_a|31, train_b|5)，逗号分隔；没有依据就写 无
        回答: 你的最终回答正文

        "证据不足"的判定标准（满足任意一条即为不足）：
        - 候选页里没有任何一页讲到了问题所问的内容；
        - 候选页内容与问题只是主题相关，但回答不了问题的具体细节；
        - 用户照片与候选页对不上，无法据此判断。
        证据不足时，回答统一写："抱歉，用户手册中没有找到相关信息，建议联系客服确认。"

        回答时不要提及"候选页""检索""图片"这类系统词汇，就像直接回答用户一样。

        对话历史: {history}
        用户问题: {question}

        输出:
        """,
        input_variables=["history", "question", "pages_desc"],
    )


# ============================================================
# 三、调用: L3 走文本模型, L4 走视觉模型
# ============================================================
class LLMClient:
    """
    函数作用: 封装 DashScope(OpenAI 兼容模式)的文本 / 视觉两条调用路径。
             纯文本任务(L3 直答)走 qwen-plus; 带图任务(L4 生成)走 qwen-vl-max。
    """

    def __init__(self, api_key=None, base_url=None, text_model=None, vl_model=None):
        # 密钥只从 .env 读(base/config.py 里 load_dotenv 已注入)。留可注入的构造参数是为了单测能塞假客户端。
        self.api_key = api_key or conf.DASHSCOPE_API_KEY
        self.base_url = base_url or conf.DASHSCOPE_BASE_URL
        self.text_model = text_model or conf.LLM_MODEL
        # '-latest' 后缀的 id 实测 403, 配置里写的就是可用 id, 这里不要自作主张加后缀
        self.vl_model = vl_model or conf.LLM_VL_MODEL
        # 延迟建连: 只 import 不调用时(如建库脚本顺带 import 本模块)不该因为密钥没配就炸
        self._client = None

    # ------------------------------------------------------------
    # 客户端
    # ------------------------------------------------------------
    @property
    def client(self):
        """惰性创建 OpenAI 客户端(同一进程内复用, 复用底层连接池)。"""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    # ------------------------------------------------------------
    # 纯文本
    # ------------------------------------------------------------
    def chat_text(self, prompt, temperature=0.1):
        """
        函数功能: 纯文本单轮问答 -> L3「通用知识」直答用。
        :param prompt: 用户提示词(已由 PromptTemplate format 好)
        :param temperature: 采样温度。直答要稳, 默认 0.1
        :return: str, 模型回答; 失败返回 ERROR_PREFIX 开头的文本(调用方据此判失败, 不抛异常)

        本函数不收 `system` / `model` 形参(全仓无人传); 要加 system 提示, 直接拼进 prompt 模板。
        """
        messages = [{'role': 'user', 'content': prompt}]

        try:
            completion = self.client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                temperature=temperature,
            )
            if not completion.choices:
                logger.warning('文本模型返回空 choices')
                return ERROR_PREFIX + '模型未返回任何内容'
            return completion.choices[0].message.content or ''
        except Exception as e:
            # 网络/鉴权/限流都在这里兜住: 级联链路上层还有一级"拒答或转人工", 不能因为它把整进程带崩
            logger.error(f'文本模型调用失败: {e}')
            return ERROR_PREFIX + f'调用大模型失败({e})'

    # ------------------------------------------------------------
    # 图文(视觉)
    # ------------------------------------------------------------
    def _build_image_parts(self, images):
        """
        函数功能: 把各种形态的图片统一成 qwen-vl 要的 content 片段列表。
        :param images: list, 元素可以是 PIL.Image / 磁盘路径(str) / 已编码的 data URL(str)
        :return: list[dict], 形如 [{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,...'}}]
        """
        parts = []
        for item in images or []:
            if item is None:
                continue

            try:
                if isinstance(item, str):
                    if item.startswith('data:'):
                        # 已经是 data URL, 直接用(前端传上来的用户实拍图走这条)
                        url = item
                    else:
                        # 磁盘路径 -> 读进来缩放再编码。候选页图走这条
                        img = Image.open(item).convert('RGB')
                        url = image_to_data_url(shrink_image(img, VL_IMAGE_MAX_SIDE))
                else:
                    # PIL.Image
                    url = image_to_data_url(shrink_image(item, VL_IMAGE_MAX_SIDE))
            except Exception as e:
                # 单张图坏掉不该废掉整轮问答: 跳过它, 用剩下的图继续答
                logger.warning(f'图片编码失败, 已跳过: {item if isinstance(item, str) else type(item)}, {e}')
                continue

            parts.append({'type': 'image_url', 'image_url': {'url': url}})

        return parts

    def chat_vl(self, prompt, images=None, temperature=0.1):
        """
        函数功能: 图文混合单轮问答 -> L4 多模态生成用。
        :param prompt: 提示词(已 format, 含"证据/页码/回答"格式要求)
        :param images: list[PIL.Image | 路径 | data URL], 顺序即送给模型的顺序
        :param temperature: 默认 0.1。手册问答要的是忠于原文, 不要发挥
        :return: str, 模型回答; 失败返回 ERROR_PREFIX 开头的文本
        """
        content = [{'type': 'text', 'text': prompt}]
        content.extend(self._build_image_parts(images))

        messages = [{'role': 'user', 'content': content}]

        try:
            completion = self.client.chat.completions.create(
                model=self.vl_model,
                messages=messages,
                temperature=temperature,
            )
            if not completion.choices:
                logger.warning('视觉模型返回空 choices')
                return ERROR_PREFIX + '视觉模型未返回任何内容'
            return completion.choices[0].message.content or ''
        except Exception as e:
            logger.error(f'视觉模型调用失败: {e}')
            return ERROR_PREFIX + f'调用视觉模型失败({e})'


# ------------------------------------------------------------
# 模块级单例: 编排器只创建一次(每次新建都重复读 .env、重复建连接池)
# ------------------------------------------------------------
_default_client = None


def get_llm_client():
    """函数功能: 取进程内共享的 LLMClient 实例。"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


if __name__ == '__main__':
    # ============================================================
    # 测试代码(仅直接运行本文件时执行)
    #   跑法: python rag_qa/core/llm_client.py
    #   ①~④ 是纯逻辑断言, **零依赖**, 冷机器上永远能跑;
    #   ⑤~⑧ 会真实出网、真实计费, 前提是 .env 里有可用的 DASHSCOPE_API_KEY。
    # ============================================================
    # ① data URL 往返: 编码后再解码, 尺寸应一致(证明 base64 链路通)
    src = Image.new('RGB', (2000, 1000), (200, 30, 30))
    url = image_to_data_url(src)
    back = load_image_from_base64(url)
    assert back.size == src.size, f'base64 往返尺寸变了: {back.size} != {src.size}'
    print(f'① base64 往返: {back.size} OK (前缀 {url[:32]}...)')

    # ② 长边缩放只缩不放
    small = shrink_image(src, 500)
    assert max(small.size) == 500, f'长边应缩到 500, 实得 {small.size}'
    tiny = shrink_image(Image.new('RGB', (10, 10)), 500)
    assert tiny.size == (10, 10), f'小图不该被放大, 实得 {tiny.size}'
    print(f'② 缩放: 2000x1000 限 500 -> {small.size}; 10x10 原样 -> {tiny.size} OK')

    # ③ 坏图 / 空输入都返回 None —— 带图问答要能降级成纯文本, 不能崩
    assert load_image_from_base64('data:image/png;base64,bm90_YW5faW1hZ2U=') is None
    assert load_image_from_base64(None) is None
    print('③ 坏图 / 空输入 -> None OK')

    # ④ 两个模板能 format, 不报 KeyError
    direct = direct_prompt().format(history='', question='什么是变速箱?')
    assert '什么是变速箱' in direct
    vl = vl_prompt().format(history='', question='电动尾门怎么开?')
    assert '证据' in vl and '页码' in vl and '电动尾门怎么开' in vl
    print('④ direct_prompt / vl_prompt 渲染 OK')

    # ---------------- 以下会出网 ----------------
    llm = LLMClient()
    print(f'⑤ 模型配置: text={llm.text_model} vl={llm.vl_model} '
          f'key={"已配置" if llm.api_key and len(llm.api_key) > 8 else "缺失"}')

    # ⑥ 文本直答
    p = direct_prompt().format(history='', question='用一句话说明什么是变速箱。')
    out = llm.chat_text(p)
    ok = bool(out) and not out.startswith(ERROR_PREFIX)
    print(f'⑥ 文本直答 {"OK" if ok else "FAIL"}: {out[:80]}')

    # ⑦ 视觉: 造一张纯色图问颜色, 确认图片真的被模型"看见"了
    red = Image.new('RGB', (300, 300), (220, 20, 20))
    out = llm.chat_vl('这张图片是什么颜色的?只回答颜色。', images=[red])
    ok = bool(out) and not out.startswith(ERROR_PREFIX) and '红' in out
    print(f'⑦ 视觉识色 {"OK" if ok else "FAIL"}: {out[:80]}')

    # ⑧ 坏图跳过不崩: 混入一个不存在的路径, 剩余图片仍应正常作答
    out = llm.chat_vl('这张图片是什么颜色的?只回答颜色。',
                      images=['/no/such/file.png', red])
    ok = bool(out) and not out.startswith(ERROR_PREFIX)
    print(f'⑧ 坏图降级 {"OK" if ok else "FAIL"}: {out[:80]}')
