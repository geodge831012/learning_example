# coding: utf-8

# python3.6 使用的是最新的keras2.2.4
# deep learning 多层神经网络

import os
import numpy as np
import keras
import jieba_fast
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from gensim.models import Word2Vec


#########################################################################
## sigmoid函数 ##
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#########################################################################
## 求X Y两个向量的cosine值
def get_cosine_value(X, Y):
    # 分子 x1*y1 + x2*y2 + ... + xn*yn
    # 分母 ||X|| * ||Y||

    if (np.linalg.norm(X) <= 0.0 or np.linalg.norm(Y) <= 0.0):
        return 0

    X = X.reshape(1, 256)
    Y = Y.reshape(1, 256)

    return float(X.dot(Y.T) / (np.linalg.norm(X) * np.linalg.norm(Y)))


##################################句子处理#######################################
class SentenceEmbedding():
    def __init__(self):

        # 停用词文件路径
        self.stopword_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/all_stopword.txt")

        # 停用词集合
        self.stopword_list = []

        # 停用词加载
        self.load_stopword()

        # word2vec模型
        self.model = Word2Vec.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../word2vec/word2vec_wx"))

        # 意图train文件路径
        self.intention_train_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/intention_train.txt")

        # 意图eva文件路径
        self.intention_eva_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/intention_eva.txt")

        # 意图test文件路径
        self.intention_test_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data/intention_test.txt")

        # 所有句子的集合 train eva test
        self.train_sentence_list    = []
        self.eva_sentence_list      = []
        self.test_sentence_list     = []



    # 加载所有的停用词 到 类变量中
    def load_stopword(self):

        for line in open(self.stopword_file):
            self.stopword_list.append(line.strip())



    # 获取某个词的word2vec值 256维
    def get_word2vec(self, word):

        if(word in self.model):
            return self.model[word]
        else:
            return np.zeros(0, dtype=np.float)


    # 获取某一个句子的句向量
    def get_sentence_embedding(self, sentence):

        # 句子的向量
        sentence_embedding = np.zeros(self.model.vector_size, dtype=np.float)

        rst = jieba_fast.cut(sentence)

        for word in rst:

            if word in self.stopword_list:
                # 停用词跳过
                continue

            # word2vec某词
            w2v_vector = self.get_word2vec(word)
            if 0 == len(w2v_vector):
                continue

            # 可以开始计算了
            coe = 1.0

            sentence_embedding += coe * w2v_vector
            #sentence_embedding = sentence_embedding + coe * w2v_vector

        return sentence_embedding


    # 处理所有的预设的句子
    def proc_intention(self, filename):

        all_sentence_list = []

        start_num = 1

        for line in open(filename):
            line_list = line.strip().split("\t")

            if( 2 != len(line_list)):
                continue

            intention   = int(line_list[0])
            sentence    = line_list[1]
            sentence_embedding = self.get_sentence_embedding(sentence)

            sentence_info_dict = {}
            sentence_info_dict["sentence_id"]           = start_num
            sentence_info_dict["sentence"]              = line
            sentence_info_dict["sentence_embedding"]    = sentence_embedding
            sentence_info_dict["intention"]             = intention

            start_num += 1

            all_sentence_list.append(sentence_info_dict)


        print("---------------------------")
        print(filename)
        print(len(all_sentence_list))

        return all_sentence_list


    #
    def proc_all_intention(self):
        self.train_sentence_list = self.proc_intention(self.intention_train_file)
        self.eva_sentence_list   = self.proc_intention(self.intention_eva_file)
        self.test_sentence_list  = self.proc_intention(self.intention_test_file)



##################################问答DNN模型#######################################
class QaDNNModel():
    def __init__(self):

        pass

    # 预处理sentence数据
    def pre_proc_sentence(self, train_sentence_list, eva_sentence_list, test_sentence_list):

        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(len(eva_sentence_list))

        self.x_train = np.empty([0, 256], dtype=np.float)
        self.y_train = np.empty([0, 19], dtype=np.float)

        for i in range(len(train_sentence_list)):
            self.x_train = np.vstack((self.x_train, train_sentence_list[i]["sentence_embedding"]))
            tmp_array = np.zeros(19)
            tmp_array[train_sentence_list[i]["intention"]-1] = 1
            self.y_train = np.vstack((self.y_train, tmp_array))

        print("===================================")
        print(type(self.x_train))
        print(self.x_train.shape)
        print(type(self.y_train))
        print(self.y_train.shape)
        print(self.y_train)
        print(get_cosine_value(self.x_train[0], self.x_train[65]))


        ##############################################################################
        self.x_eva = np.empty([0, 256], dtype=np.float)
        self.y_eva = np.empty([0, 19], dtype=np.float)

        for i in range(len(eva_sentence_list)):
            self.x_eva = np.vstack((self.x_eva, eva_sentence_list[i]["sentence_embedding"]))
            tmp_array = np.zeros(19)
            tmp_array[eva_sentence_list[i]["intention"] - 1] = 1
            self.y_eva = np.vstack((self.y_eva, tmp_array))


        print("===================================")
        print(type(self.x_eva))
        print(self.x_eva.shape)
        print(type(self.y_eva))
        print(self.y_eva.shape)
        print(self.y_eva)


        ##############################################################################
        self.x_test = np.empty([0, 256], dtype=np.float)

        for i in range(len(test_sentence_list)):
            self.x_test = np.vstack((self.x_test, test_sentence_list[i]["sentence_embedding"]))




    # 构建deep learning模型
    def build_dnn_model(self):

        # build a model
        self.model = Sequential()

        # fill model, the first layer
        # 第一层上有500个neural节点 输入是20维的 具体数据可以随意 本例子是1000个（相当于1000个样本 每个样本20个维度）
        #self.model.add(Dense(units=500, activation='sigmoid', input_dim=256))
        self.model.add(Dense(units=500, activation='relu', input_dim=256))
        self.model.add(Dropout(0.5))

        # fill model, the second layer
        # 第二层上有500个neural节点
        #self.model.add(Dense(units=500, activation='sigmoid'))
        for i in range(3):
            self.model.add(Dense(units=500, activation='relu'))
            self.model.add(Dropout(0.5))

        # fill model, the output
        # 最终输出是10维的，就是每个样本在10个维度上的flag 0 or 1，相当于是一个分类问题
        self.model.add(Dense(units=19, activation='softmax'))

        # model compile
        print("========compile begin========")
        #self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
        print("========compile end========")

        print("========fit begin, real train========")
        self.model.fit(self.x_train, self.y_train, batch_size=10, nb_epoch=20)
        print("========fit end========")

        # evaluate
        score = self.model.evaluate(self.x_eva, self.y_eva)
        print(score)

        # predict
        result = self.model.predict(self.x_test)
        print(100.0 * result)



#########################################################################
## main 主函数 ##

if __name__ == '__main__':

    sentence_embedding = SentenceEmbedding()

    sentence_embedding.proc_all_intention()

    qa_dnn_model = QaDNNModel()

    qa_dnn_model.pre_proc_sentence(sentence_embedding.train_sentence_list, sentence_embedding.eva_sentence_list, sentence_embedding.test_sentence_list)

    qa_dnn_model.build_dnn_model()


