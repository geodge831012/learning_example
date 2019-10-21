#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import datetime
import json
import traceback
from gensim import corpora, models, similarities
import pymysql
import jieba
import re
import os
from utils.file_util import *


def etl(content):
    etlregex = re.compile(u"[^\u4e00-\u9fa5a-zA-Z0-9]")
    content = etlregex.sub('',content)
    return content

#读出所有的文件内容
def read_all_file():
    id_list = []
    content_list = []

    file_list = foreach_filepath('news/')
    for i in range(len(file_list)):
        id_str = os.path.basename(file_list[i])
        content_str = read_file_content(file_list[i])

        id_list.append(id_str)
        content_list.append(content_str)

    return id_list, content_list


def train_model():
    print("begin train_model...")

    id_list, article_arr = read_all_file()

    #for i in range(len(id_list)):
    #    print(i)
    #    print(id_list[i])
        #print(article_arr[i])
    #    print("==================")
#    return

    # 原始语料集合
    train_set = []
    docinfos = []
    # 读取文本，进行切词操作
    for i in range(len(article_arr)):
        content_str = article_arr[i]

        word_list = []
        for word in jieba.cut(content_str):
            word_list.append(word)
        train_set.append(word_list)

    dictionary = corpora.Dictionary(train_set)
    dictionary.filter_extremes(no_below=1, no_above=1, keep_n=None)
    #for w in dictionary:
    #    print(w)
    #    print(dictionary[w])
    corpus = [dictionary.doc2bow(text) for text in train_set]
    print("===========corpus==============")
    #for i in range(len(corpus)):
    #    print(i)
    #    print(corpus[i])
    #    print("+++++++++++++++++++++++++++++++++")

    # corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
    tfidfModel = models.TfidfModel(corpus)
    tfidfVectors = tfidfModel[corpus]

    #查看model中的内容
    for item in tfidfVectors:
        print(item)
        print(len(item))
    print("++++++++++++++++++++++++++++++++++++++")



    #indexTfidf = similarities.MatrixSimilarity(tfidfVectors)
    indexTfidf = similarities.SparseMatrixSimilarity(tfidfVectors, num_features=8)
    #indexTfidf = similarities.Similarity('Similarity-tfidf-index', corpus_tfidf, num_features=600)

    print("hhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    #for i in indexTfidf:
    #    print(i)
    #    print(len(i))
    #print(indexTfidf)
    #print(len(indexTfidf))


    print("end train_model...")

    print("start save model...")
    try:
        dictionary.save("model\\all.dic")
        #text_util.saveObject(output + "all.info", docinfos)
        tfidfModel.save("model\\allTFIDF.mdl")
        indexTfidf.save("model\\allTFIDF.idx")
    except:
        print("save model exception")
    print("end save model...")


#预测文章
def tfidf_predict():
    print("tfidf_predict start...")

    content_str = read_file_content('news/2751104')
    print("content_str:" + content_str)

    print("load model start...")
    try:
        #docinfos = text_util.loadObject(output + "all.info")  # 载入详情数据
        dictionary = corpora.Dictionary.load("model\\all.dic")  # 载入字典
        tfidfModel = models.TfidfModel.load("model\\allTFIDF.mdl")  # 载入TFIDF模型
        indexTfidf = similarities.MatrixSimilarity.load("model\\allTFIDF.idx")  # 载入相似模型
    except:
        print("load model exception")
        return
    print("load model end...")

    print(dictionary)

    query_bow = dictionary.doc2bow(filter(lambda x: len(x) > 0, map(etl, jieba.cut(content_str))))
    print(query_bow)
    tfidfvect = tfidfModel[query_bow]
    simstfidf = indexTfidf[tfidfvect]
    sort_sims = sorted(enumerate(simstfidf), key=lambda item: -item[1])
    #article['related_news'] = []

    for sim in sort_sims:  # sim[0]是文本的index，sim[1]是相似度的值
        doc_index = sim[0]
        similar_value = sim[1]
        #print("doc_index:%s"%format(doc_index))
        print("doc_index:%s"%format(doc_index))
        print("similar_value:%s"%format(similar_value))

        #doc_info = docinfos[doc_index]


if __name__ == '__main__':
    print("main start")

    train_model()
    #tfidf_predict()

    print("main end")

    # remove_duplicate_report()
    # article = {'title': "test",
    #            'content': u'''以视频为核心的物联网解决方案和数据运营服务提供商海康威视宣布，计划在蒙特利尔建立研发中心，在硅谷建立研究所。这是海康威视首次在中国境外设立研发机构。
    #  预计在2017年开始运营的海康威视蒙特利尔研发中心将专注於工程研发。海康威视硅谷研究所则将专注於广泛的技术研究。蒙特利尔拥有卓越的人才库和适合企业发展的良好环境，是北美区域新研发中心的理想位置。作为高科技聚集地的硅谷则是设立海康威视研究所的最佳选择。
    # 海康威视总裁胡扬忠表示，蒙特利尔研发中心和和硅谷研究所的设立，将进一步提升海康威视的研发实力，同时有助於提升中国以外地区的本地支持与服务。
    #  其中，蒙特利尔研发中心将更好地践行为北美地区重要集成商合作夥伴提供定制化企业级产品的承诺。在过去几年，海康威视在美国和加拿大打造了具备出色专业知识的工程、技术和销售团队，为客户提供企业解决方案服务。「海康威视深知不同地区对解决方案的需求各不相同。新的海康威视研发团队将致力於开发专门为北美企业市场而设计的新产品。」海康威视美国公司及海康威视加拿大公司总经理Jeffrey He说道。
    # 总部位於中国杭州的海康威视拥有业内突出的研发能力，现拥有8,000多名研发工程师，并且每年将营收的7%左右投入研发工作，在研发上的持续投入为公司的良好发展提供了强大的技术支撑。'''}
    # tfidf_predict(article)
    #
    # # load_reports.get_reports_by_reportTime('1535098912')
    # res= get_timestamp('1535098912')
    # print res
