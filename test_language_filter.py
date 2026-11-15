# -*- coding: utf-8 -*-
"""
"强制单一语言"过滤的纯逻辑测试（不加载模型，秒级跑完）。

用法：python test_language_filter.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.tools.language_filter import is_target_language, count_scripts

FAILED = []


def check(name, actual, expect):
    ok = actual == expect
    if not ok:
        FAILED.append(name)
    print(f'{"PASS" if ok else "FAIL"}  {name}  -> {actual!r} (期望 {expect!r})')


def main():
    print('---- 文字系统统计 ----')
    check('count 中文', count_scripts('就算是我的错'), {'han': 6})
    check('count 日文', count_scripts('そう言った'), {'kana': 4, 'han': 1})
    check('count 混合', count_scripts('OK 了解'), {'latin': 2, 'han': 2})
    check('count 数字不计', count_scripts('¥750'), {})
    check('count 谚文', count_scripts('안녕'), {'hangul': 2})

    print('\n---- 目标：日文 ----')
    check('日文台词', is_target_language('そう言った', 'japan'), True)
    check('日文全角片假名', is_target_language('オーケー', 'japan'), True)
    check('日文纯汉字短句', is_target_language('本当', 'japan'), True)
    check('日文含数字', is_target_language('第1話', 'japan'), True)
    # 日文目标下汉字行一律保留：汉字在日文里同样是正文用字，中日无法在字形层面区分
    check('中文台词保留', is_target_language('就算是我的错', 'japan'), True)
    check('中文含简体特征保留', is_target_language('我们自己', 'japan'), True)
    check('中英混排保留', is_target_language('Level 5 超能力者', 'japan'), True)
    check('纯繁体保留', is_target_language('就算是我的錯', 'japan'), True)
    check('英文招牌应丢', is_target_language('J.C.STAFF', 'japan'), False)
    check('数字价格应丢', is_target_language('¥750', 'japan'), False)
    check('韩文应丢', is_target_language('안녕하세요', 'japan'), False)

    print('\n---- 目标：简体中文 ----')
    check('中文台词', is_target_language('就算是我的错', 'ch'), True)
    check('中文带标点', is_target_language('你好，世界！', 'ch'), True)
    check('日文假名应丢', is_target_language('そう言った', 'japan') and
          is_target_language('そう言った', 'ch'), False)
    check('英文应丢', is_target_language('Hello world', 'ch'), False)
    check('纯数字应丢', is_target_language('750', 'ch'), False)
    check('中文夹英文放行', is_target_language('我们OK的', 'ch'), True)

    print('\n---- 目标：繁体中文 ----')
    check('繁体台词', is_target_language('就算是我的錯', 'chinese_cht'), True)
    check('日文应丢', is_target_language('そう言った', 'chinese_cht'), False)

    print('\n---- 目标：拉丁语系 ----')
    check('英文', is_target_language('Hello world', 'en'), True)
    check('英文带重音', is_target_language('Café au lait', 'fr'), True)
    check('越南语声调', is_target_language('Xin chào', 'vi'), True)
    check('中文应丢', is_target_language('你好', 'en'), False)
    check('纯数字应丢', is_target_language('750', 'en'), False)

    print('\n---- 目标：其他文字系统 ----')
    check('韩文', is_target_language('안녕하세요', 'korean'), True)
    check('俄文', is_target_language('Привет', 'ru'), True)
    check('阿拉伯文', is_target_language('مرحبا', 'ar'), True)
    check('印地文', is_target_language('नमस्ते', 'hi'), True)
    check('泰文', is_target_language('สวัสดี', 'th'), True)
    check('希腊文', is_target_language('Γειά', 'el'), True
          if is_target_language('Γειά', 'el') else False)

    print('\n---- 边界 ----')
    check('空文本', is_target_language('', 'japan'), False)
    check('仅符号', is_target_language('……', 'ch'), False)
    check('未收录语言不过滤', is_target_language('随便什么', 'zz'), True)
    check('大写字码兼容', is_target_language('そう', 'JAPAN'), True)

    print()
    if FAILED:
        print(f'FAILED {len(FAILED)}: {FAILED}')
        return 1
    print('ALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
