# coding:utf-8

import re
import jieba
from simhash import Simhash


# 字符串过滤
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


def proc_simhash(text):

    t1 = cleanhtml(text)
    t2 = cleanhtml(text)

    simhash_result1 = Simhash(jieba.lcut(t1))
    simhash_result2 = Simhash(jieba.lcut(t2))

    print("%x" % simhash_result1.value)
    print(simhash_result1.distance(simhash_result2))




if __name__ == '__main__':

    text = '在此次活动上，王小川展示了搜狗推出的全球首个高度实用化、定制化的虚拟主播。结合唇语合成、语音合成、音视频联合建模与深度学习技术，可驱动机器生成对应的唇语图像与声音，进而输出统一的音视频素材。此次展示的虚拟主播，是搜狗使用央视新闻主播姚雪松数十个小时的音视频素材进行训练与计算的结果，最终生成了一段音视频同步的RISE大会新闻播报，与真人播报无异。'

    proc_simhash(text)






