# -*- coding: utf-8 -*-

from xpinyin import Pinyin
p = Pinyin()

a = p.get_pinyin(u"上海")
print(a)
print(type(a))

b = p.get_pinyin(u"上海", tone_marks='marks')
print(b)
print(type(b))

c = p.get_pinyin(u"上海", '')
print(c)
print(type(c))

