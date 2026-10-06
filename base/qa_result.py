# 级联各层的统一返回协议 QAResult + 缓存 key 归一化(cache_key): 编排器只判 result.hit
# (放 base/ 是因为 mysql_qa 与 new_main 都要用, 不该让 mysql_qa 反向依赖 rag_qa)

import hashlib
import unicodedata          # 用 Unicode 分类判标点, 比手写标点表全
from dataclasses import dataclass, field


@dataclass
class QAResult:
    """
    一处问答结果 —— 级联每一层的唯一返回形态。

    :param answer: 答案正文; 拒答时是拒答话术(不是空串, 前端要直接显示)
    :param source: 来源取值 redis/mysql/direct/rag/refuse, 兼作命中率统计分组键
    :param hit:    False = 编排器继续往下一层丢。注意 refuse 是 hit=True ——
                   拒答是 L4 的正常产出, 不是"没命中"
    :param meta:   中间产物: 延迟、BM25 分数、检索 top 页码、召回页数等
    """
    answer: str
    source: str
    hit: bool
    meta: dict = field(default_factory=dict)

    # 便捷构造: 调用方不用记 source 字符串字面量, 少一处拼错机会
    @classmethod
    def redis_hit(cls, answer, **meta):
        return cls(answer=answer, source='redis', hit=True, meta=meta)

    @classmethod
    def mysql_hit(cls, answer, **meta):
        return cls(answer=answer, source='mysql', hit=True, meta=meta)

    @classmethod
    def direct_answer(cls, answer, **meta):
        """L3 判为「通用知识」, qwen-plus 直答(不检索)"""
        return cls(answer=answer, source='direct', hit=True, meta=meta)

    @classmethod
    def rag_answer(cls, answer, **meta):
        """L4 多模态 RAG 生成"""
        return cls(answer=answer, source='rag', hit=True, meta=meta)

    @classmethod
    def refuse(cls, answer, **meta):
        """
        拒答 —— L4 的一等公民, 必须显式拒答而非编造:
        测试集混有手册外问题 + "像专业问题但手册里没有"的盲区(BERT 只分通用/专业,
        分不出"手册里有没有")。hit=True, 编排器不再往下丢。
        """
        return cls(answer=answer, source='refuse', hit=True, meta=meta)

    @classmethod
    def miss(cls, source, **meta):
        """本层未命中, 编排器继续下一层"""
        return cls(answer='', source=source, hit=False, meta=meta)


# 进 L1 前必须先归一化, 否则同句只是标点/空格不同就缓存不中 —— 命中率虚低。
# 全角(FF01~FF5E)与半角(21~7E)码位差, 固定值 0xFEE0
_FULLWIDTH_OFFSET = 0xFEE0


def _to_halfwidth(text):
    """全角转半角: 『ＡＢ，１』-> 『AB,1』; 全角空格(U+3000)转普通空格。"""
    chars = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:          # 全角 ASCII 区
            chars.append(chr(code - _FULLWIDTH_OFFSET))
        elif code == 0x3000:                  # 全角空格
            chars.append(' ')
        else:
            chars.append(ch)
    return ''.join(chars)


def _strip_punct_and_space(text):
    """
    去掉所有标点与空白。用 unicodedata.category 判: P* = Punctuation, Z* = Separator ——
    手写标点表必漏(中英文标点/全角引号/书名号/省略号…)。
    """
    return ''.join(ch for ch in text
                   if not unicodedata.category(ch).startswith(('P', 'Z')))


def normalize_query(query):
    """
    把用户问题归一化成 L1 缓存 key 的原料。
    顺序: 去首尾空格 -> 全角转半角 -> 转小写 -> 去标点与空格
    """
    if not query:
        return ''
    text = str(query).strip()
    text = _to_halfwidth(text)
    text = text.lower()
    return _strip_punct_and_space(text)


def cache_key(query):
    """
    造 L1 缓存 key, 形如 "qa:<md5>"。纯文本归一化后取 md5。
    """
    normalized = normalize_query(query)
    return 'qa:' + hashlib.md5(normalized.encode('utf-8')).hexdigest()


if __name__ == '__main__':
    # 冒烟自测: python base/qa_result.py
    pairs = [
        ('怎样加热座椅?', '怎样加热座椅'),          # 半角问号
        ('怎样加热座椅？', '怎样加热座椅'),          # 全角问号 U+FF1F
        ('  怎样加热座椅  ', '怎样加热座椅'),        # 首尾空格
        ('How to heat the seat?', 'howtoheattheseat'),  # 英文大小写 + 标点
        ('电动尾门，怎么打开！', '电动尾门怎么打开'),  # 中文标点
        ('ＡＢＣ １２３', 'abc123'),                 # 全角字母数字 + 空格
    ]
    for src, expect in pairs:
        got = normalize_query(src)
        print(f'① 归一化 {src!r} -> {got!r} {"OK" if got == expect else f"FAIL 期望 {expect!r}"}')

    # 同句不同写法必须同 key(L1 命中率关键)
    k1, k2 = cache_key('怎样加热座椅?'), cache_key('  怎样加热座椅  ')
    print(f'② 不同写法同 key: {k1} == {k2} -> {k1 == k2}')

    print(f'③ 不同问题不同 key: {cache_key("座椅加热") != cache_key("座椅通风")}')

    cases = [
        QAResult.redis_hit('a'), QAResult.mysql_hit('b'), QAResult.direct_answer('c'),
        QAResult.rag_answer('d'), QAResult.refuse('e'), QAResult.miss('l1'),
    ]
    print('⑤ 协议构造:', [(r.source, r.hit) for r in cases])
    print('⑤ 期望     :', [('redis', True), ('mysql', True), ('direct', True),
                            ('rag', True), ('refuse', True), ('l1', False)])

    
    # 纯文本 key 逐位稳定(它是 L1/L2 共用, 改坏会让纯文本缓存失效且不报错)
    assert cache_key('怎样加热座椅?') == cache_key('  怎样加热座椅  '), '归一化被破坏了'
    assert cache_key('怎样加热座椅') == 'qa:' + hashlib.md5('怎样加热座椅'.encode('utf-8')).hexdigest()
    print('⑥ 纯文本 key 逐位未变 OK')