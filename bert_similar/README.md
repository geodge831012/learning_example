# bert_similar

使用BERT计算句子的向量

使用BERT模型最后一层的值作为输入句子的向量

再使用夹角余弦求两个句子的相似度


参考链接:

https://juejin.im/post/5e6ce1426fb9a07cdc60164c

https://github.com/hanxiao/bert-as-service



需要安装的安装包:
/home/mqq/geodge/anaconda3/bin/pip install -U bert-serving-server bert-serving-client

启动服务
/home/mqq/geodge/anaconda3/bin/bert-serving-start -model_dir /home/mqq/geodge/BERT/model/chinese_L-12_H-768_A-12 -num_worker=4

调用客户端
/home/mqq/geodge/anaconda3/bin/python similar.py
