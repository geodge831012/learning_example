# pinyin
汉字转拼音

pip install xpinyin

https://www.jb51.net/article/167461.htm

 >>> from xpinyin import Pinyin
 >>> p = Pinyin()
 >>> # default splitter is `-`
 >>> p.get_pinyin(u"上海")
 'shang-hai'
 >>> # show tone marks
 >>> p.get_pinyin(u"上海", tone_marks='marks')
 'shàng-hǎi'
 >>> p.get_pinyin(u"上海", tone_marks='numbers')
 >>> 'shang4-hai3'
 >>> # remove splitter
 >>> p.get_pinyin(u"上海", '')
 'shanghai'
 >>> # set splitter as whitespace
 >>> p.get_pinyin(u"上海", ' ')
 'shang hai'
 >>> p.get_initial(u"上")
 'S'
 >>> p.get_initials(u"上海")
 'S-H'
 >>> p.get_initials(u"上海", u'')
 'SH'
 >>> p.get_initials(u"上海", u' ')
 'S H'
 
 请输入utf8编码汉字
 _chinese\_pinyin: https://github.com/flyerhzm/chinese_pinyin
