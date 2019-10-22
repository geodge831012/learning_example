# keras_learning
使用keras来做的例子


keras_dl.py

python3.6 使用的是最新的keras2.2.4

deep learning 多层神经网络


src/keras_qa_robot.p

dnn处理智能语义机器人

成功率86% 看起来不是很高



src/keras_cnn_robot.py

nlp的cnn模型，用于处理句子的cnn模型，使用的是1D卷积模型，一个句子转换成为一个2维的tensor，其中一个维度是word2vec的dimension，另外一个维度是词汇的个数，如果太长的句子，就截断，如果太短，则补0

论文见../doc/text_cnn_classification.pdf  博客见https://blog.csdn.net/qq_25037903/article/details/85058217

成功率80%+ 看起来也不是很高



src/picture_cnn.py

picture的cnn模型，使用的是2D卷积模型，数据源是从keras里面自带获取的
