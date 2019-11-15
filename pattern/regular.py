# -*- coding: utf-8 -*-

import re

pattern = re.compile(u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')

doc = "世界你好st海康大涨"
doc_key = pattern.sub(u'', doc)
print(doc)
print(doc_key)
