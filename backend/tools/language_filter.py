# -*- coding: utf-8 -*-
"""
按指定语言过滤 OCR 识别结果（"强制单一语言"功能）。

背景：字幕选区里除了台词，经常还有招牌、台标、道具文字、片头 logo 等。
这些内容一旦被识别成文本，就会变成 srt 里的垃圾条目（实测 S01E03 有
'¥750'、'J.C.STAFF'、'自動販売機7116号' 这类）。语言过滤按**文字系统**
判断一行是不是目标语言，不是就直接丢弃。

判定方式：统计文本里各文字系统（汉字 / 假名 / 谚文 / 拉丁 / 西里尔 ...）的字符数，
再看目标语言需要哪几种：
  - 整行没有任何文字（只有数字和符号，如 '750'、'¥750'、'！？'）→ **一律保留**，
    数字与符号是语言无关的，任何语言下都不该被语言过滤丢掉
  - 有文字但需要的系统一个字符都没有 → 丢弃（日文目标丢掉纯英文招牌）
  - 有明显的冲突系统（中文目标遇到假名）→ 丢弃

日文目标**不区分中日汉字**：日文正文本来就是汉字+假名混排，纯汉字行
（「本当」「大丈夫」）是合法台词，中文行也一样保留——两者在字形层面无法可靠
区分，误杀台词的代价远大于多留几行中文。所以日文目标只过滤掉完全不含
汉字/假名的内容（英文台标、数字价格、韩文等）。

已知局限：中文目标能靠「假名数量是否超过汉字」识别日文行，反过来做不到。
"""

# ---------------------------------------------------------------------------
# Unicode 区段 → 文字系统
# ---------------------------------------------------------------------------
# 只列实际可能出现在字幕里的区段，够用且判定快（每行几十个字符，纯 O(1) 查表）
_SCRIPT_RANGES = {
    'han': ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x3005, 0x3005)),
    'kana': ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)),
    'hangul': ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7FF)),
    'latin': ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F),
              (0x1E00, 0x1EFF), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)),
    'cyrillic': ((0x0400, 0x04FF), (0x0500, 0x052F)),
    'arabic': ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
               (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    'devanagari': ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    'thai': ((0x0E00, 0x0E7F),),
    'greek': ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    'tamil': ((0x0B80, 0x0BFF),),
    'telugu': ((0x0C00, 0x0C7F),),
    'kannada': ((0x0C80, 0x0CFF),),
    'georgian': ((0x10A0, 0x10FF),),
    'hebrew': ((0x0590, 0x05FF),),
}


def _script_of(ch):
    """返回单个字符所属的文字系统，不属于任何已知系统时返回 None"""
    cp = ord(ch)
    for name, ranges in _SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def count_scripts(text):
    """统计文本中各文字系统的字符数（空格、标点、数字不计）"""
    counts = {}
    for ch in text:
        name = _script_of(ch)
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 语言 → 需要的文字系统
# ---------------------------------------------------------------------------
_LATIN_LANGS = (
    'en', 'french', 'fr', 'german', 'de', 'es', 'pt', 'it', 'af', 'az', 'bs',
    'cs', 'cy', 'da', 'et', 'ga', 'hr', 'hu', 'id', 'is', 'ku', 'la', 'lt',
    'lv', 'mi', 'ms', 'mt', 'nl', 'no', 'oc', 'pi', 'pl', 'ro', 'rs_latin',
    'sk', 'sl', 'sq', 'sv', 'sw', 'tl', 'tr', 'uz', 'vi', 'latin',
)
_CYRILLIC_LANGS = (
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic',
)
_ARABIC_LANGS = ('ar', 'fa', 'ur', 'ug')
_DEVANAGARI_LANGS = (
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc', 'devanagari',
)
_HANGUL_LANGS = ('korean', 'ko')
_KANNADA_LANGS = ('kn', 'ka')

LANG_SCRIPTS = {'ch': ('han',), 'chinese_cht': ('han',), 'japan': ('kana', 'han')}
LANG_SCRIPTS.update({k: ('latin',) for k in _LATIN_LANGS})
LANG_SCRIPTS.update({k: ('cyrillic',) for k in _CYRILLIC_LANGS})
LANG_SCRIPTS.update({k: ('arabic',) for k in _ARABIC_LANGS})
LANG_SCRIPTS.update({k: ('devanagari',) for k in _DEVANAGARI_LANGS})
LANG_SCRIPTS.update({k: ('hangul',) for k in _HANGUL_LANGS})
LANG_SCRIPTS.update({k: ('kannada',) for k in _KANNADA_LANGS})
LANG_SCRIPTS.update({
    'thai': ('thai',),
    'el': ('greek',),
    'ta': ('tamil',),
    'te': ('telugu',),
})

# 中文目标下，假名是"这行其实是日文"的强证据。
# 日文目标不设排除项：汉字在日文里同样是正文用字，不能因此判为外文。
LANG_EXCLUDE = {'ch': ('kana',), 'chinese_cht': ('kana',)}


def is_target_language(text, lang):
    """
    判断一条识别结果是否符合指定语言。

    lang 为 config.language 的取值（ch / chinese_cht / japan / en / korean ...）。
    未收录的语言一律返回 True（不过滤），避免误删。
    """
    lang = (lang or '').strip().lower()
    required = LANG_SCRIPTS.get(lang)
    if not required:
        return True

    text = (text or '').strip()
    if not text:
        return False

    counts = count_scripts(text)
    if not counts:
        # 整行没有任何文字（只有数字和符号，如 '750'、'¥750'、'！？'）：
        # 数字与符号是语言无关的，任何语言下都应该保留，不参与语言判定。
        # 真正的垃圾行（'1  1'、'...'、单字符）由噪声过滤处理，不在这里丢。
        return True

    matched = sum(counts.get(s, 0) for s in required)
    if matched == 0:
        # 有文字但一个目标文字系统的字符都没有：纯英文招牌、韩文行等
        return False

    excluded = LANG_EXCLUDE.get(lang)
    if excluded:
        conflict = sum(counts.get(s, 0) for s in excluded)
        if conflict > matched:
            # 假名比汉字还多 → 这行是日文
            return False

    # 日文目标到此为止：汉字、假名都算日文用字，不再按简体中文特征二次过滤，
    # 避免把中文台词或混排行误判成外文丢掉。
    return True
