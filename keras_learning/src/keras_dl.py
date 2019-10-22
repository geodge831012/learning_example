# coding: utf-8

# python3.6 使用的是最新的keras2.2.4
# deep learning 多层神经网络

import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


#########################################################################
## sigmoid函数 ##
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#########################################################################
## main 主函数 ##

if __name__ == '__main__':

    print("=========keras_deep_learning begin==========")

    # build a model
    model = Sequential()

    # fill model, the first layer
    # 第一层上有500个neural节点 输入是20维的 具体数据可以随意 本例子是1000个（相当于1000个样本 每个样本20个维度）
    model.add( Dense(units=500, activation='sigmoid', input_dim=20) )
    model.add(Dropout(0.5))

    # fill model, the second layer
    # 第二层上有500个neural节点
    model.add( Dense(units=500, activation='sigmoid') )
    model.add(Dropout(0.5))

    # fill model, the output
    # 最终输出是10维的，就是每个样本在10个维度上的flag 0 or 1，相当于是一个分类问题
    model.add( Dense(units=10, activation='softmax') )

    # model compile
    print("========compile begin========")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("========compile end========")

    # model fit
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    print("========fit begin, real train========")
    model.fit(x_train, y_train, batch_size=100, nb_epoch=20)
    print("========fit end========")

    # evaluate
    #score = model.evaluate(x_test, y_test, batch_size=128)
    score = model.evaluate(x_test, y_test)
    print(score)

    # predict
    result = model.predict(x_test)
    print(result)

    print("=========keras_deep_learning end==========")


