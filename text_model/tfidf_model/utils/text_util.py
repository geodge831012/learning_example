# coding:utf-8
import pickle
import re
from global_config import *


def saveObject(filename, obj):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    return True


def loadObject(filename):
    f=open(filename,'r')
    obj=pickle.load(f)
    return obj


def etl(content):
    etlregex = re.compile(ur"[^\u4e00-\u9fa5a-zA-Z0-9]")
    content = etlregex.sub('',content)
    return content


class LoadCorpora(object):
    def __init__(self, s):
        self.path = s

    def __iter__(self):
        f = open(self.path,'r')
        for news in f:
            yield news.split(' ')