import os

LTP_DATA_DIR='/home/geodge/software/ltp_data_v3.4.0'

#sentence = '熊高雄你吃饭了吗'
#sentence = '屠呦呦，女，汉族，中共党员，药学家，1930年12月30日生于浙江宁波。'
#sentence = '蒂芙尼：夏季销售额明显下滑。第二季度，知名珠宝制造商蒂芙尼（Tiffany）销售额为10亿美元，同比下降3%，低于分析师预期的11亿美元；可比同店销售额下降3%，降幅超过分析师预计的1.3%。不过，经调整每股收益为1.17美元，超过分析师预期的1.04美元。此外，蒂芙尼维持了低个位数增长的全年销售额增速指引。'
#sentence = '瑞典银行：面临洗钱丑闻 聘任新的CEO。瑞典银行（Swedbank AB）聘请银行业和金融业资深人士Jens Henriksson担任首席执行官，以重建外界对该公司的信任。此前瑞典银行爆出洗钱丑闻，导致其前首席执行官于今年3月被免职。'
sentence = '苹果：加强Siri录音隐私规定。苹果公司表示，将不再自动保留用户与语音助手之间互动的音频记录，这是苹果公司试图缓解用户隐私担忧的最新举措。'


from pyltp import SentenceSplitter
sents = SentenceSplitter.split(sentence)  # 分句
print('\n'.join(sents))
print("=========================================================")


from pyltp import Segmentor
cws_model_path = os.path.join(LTP_DATA_DIR,'cws.model')
segmentor = Segmentor()
segmentor.load(cws_model_path)
#segmentor.load_with_lexicon(cws_model_path, 'self_dict.txt') # 加载模型，第二个参数是您的外部词典文件路径
words = segmentor.segment(sentence) #分词
print('\t'.join(words))
segmentor.release()
print("=========================================================")


from pyltp import Postagger
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型

postags = postagger.postag(words)  # 词性标注
print('\t'.join(postags))
postagger.release()
print("=========================================================")


from pyltp import NamedEntityRecognizer
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
recognizer = NamedEntityRecognizer() # 初始化实例
recognizer.load(ner_model_path)  # 加载模型

netags = recognizer.recognize(words, postags)  # 命名实体识别
print('\t'.join(netags))
recognizer.release()


