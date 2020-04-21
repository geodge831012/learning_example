import pandas as pd
import numpy as np
from bert_serving.client import BertClient
from termcolor import colored

num = 100     # 采样数
topk = 5     # 返回 topk 个结果

# 读取数据集
#sentence_csv = pd.read_csv('atec_nlp_sim_train_all.csv', sep='\t', names=['idx', 's1', 's2', 'label'])
#sentences = sentence_csv['s1'].tolist()[:num]
#print('%d questions loaded, avg.len %d' % (len(sentences), np.mean([len(d) for d in sentences])))

sentences = []

for line in open('question.txt', 'r'):
    line = line.strip()
    if(len(line) > 0):
        sentences.append(line)

    # 句子太多了 计算向量处理时间太久了 因此缩短一点
    if(len(sentences) > 1000):
        break


with BertClient(port=5555, port_out=5556) as bc:

    # 获取句子向量编码
    doc_vecs = bc.encode(sentences)
#    print("====================================")
#    print(len(sentences))
#    print(len(doc_vecs))
#    print(type(doc_vecs[0]))
#    print(len(doc_vecs[0]))
#    print(doc_vecs[0])
#    print("====================================")

    while True:
        query = input(colored('your question：', 'green'))
        query_vec = bc.encode([query])[0]

        # 余弦相似度 分数计算。
        # np.linalg.norm 是求取向量的二范数，也就是向量长度。
        score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
#        print("==================score==================")
#        print(type(score))
#        print(len(score))
#        print(score)
#        print("==================score==================")
        
        '''
                argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
        
            [::-1]取从后向前（相反）的元素, 例如[ 1 2 3 4 5 ]
            则输出为[ 5 4 3 2 1 ]
        '''
        topk_idx = np.argsort(score)[::-1][:topk]
        print('top %d questions similar to "%s"' % (topk, colored(query, 'green')))
        for idx in topk_idx:
            print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(sentences[idx], 'yellow')))

