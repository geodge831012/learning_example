# coding: utf-8

# python3.6 使用的是最新的keras2.2.4
# lstm, long short-term memory长短期记忆的RNN

import os, json
import numpy as np
from tqdm import tqdm
import math
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD
from keras.callbacks import Callback

import seq2seq
from seq2seq.models import SimpleSeq2Seq

# https://blog.csdn.net/churximi/article/details/61210129
# https://zhuanlan.zhihu.com/p/39884984

min_count = 32
maxlen = 1500
batch_size = 128
epochs = 100
char_size = 128


#data_path = 'abstarct_content/train.tsv'
data_path = 'abstarct_content/tmp'
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

if os.path.exists('seq2seq_config.json'):
    chars, id2char, char2id = json.load(open('seq2seq_config.json'))
    id2char = {int(i): j for i, j in id2char.items()}
else:
    chars = {}
    for line in tqdm(lines):
        try:
            title = line.split('\t')[0]
            content = line.split('\t')[1]
            for w in content:  # 纯文本，不用分词
                chars[w] = chars.get(w, 0) + 1
            for w in title:  # 纯文本，不用分词
                chars[w] = chars.get(w, 0) + 1
        except:
            print(line)
    chars = {i: j for i, j in chars.items() if j >= min_count}
    # 0: mask
    # 1: unk
    # 2: start
    # 3: end
    id2char = {i + 4: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    json.dump([chars, id2char, char2id], open('seq2seq_config.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end:  # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen - 2]]
        ids = [2] + ids + [3]
    else:  # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    # padding至batch内的最大长度
    #ml = max([len(i) for i in x])
    ml = 1500
    return [i + [0] * (ml - len(i)) for i in x]


def data_generator():
    # 数据生成器
    X, Y = [], []
    while True:
        for line in lines:
            title = line.split('\t')[0]
            content = line.split('\t')[1]
            X.append(str2id(content))
            Y.append(str2id(title, start_end=True))
            print("X")
            #print(X)
            print(len(X))
            print(len(X[0]))

            print("Y")
            #print(Y)
            print(len(Y))
            print(len(Y[0]))

            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None
                X, Y = [], []


#########################################################################
## main 主函数 ##

if __name__ == '__main__':

    print("=========keras_lstm begin==========")


    def gen_title(s, topk=3):
        """beam search解码
        每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
        """
        xid = np.array([str2id(s)] * topk)  # 输入转id
        yid = np.array([[2]] * topk)  # 解码均以<start>开通，这里<start>的id为2
        scores = [0] * topk  # 候选答案分数
        for i in range(100):  # 强制要求标题不超过50字
            proba = model.predict([xid, yid])[:, i, 3:]  # 直接忽略<padding>、<unk>、<start>
            log_proba = np.log(proba + 1e-6)  # 取对数，方便计算
            arg_topk = log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
            _yid = []  # 暂存的候选目标序列
            _scores = []  # 暂存的候选目标序列得分
            if i == 0:
                for j in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[0][j] + 3])
                    _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
            else:
                for j in range(len(xid)):
                    for k in range(topk):  # 遍历topk*topk的组合
                        _yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                        _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
                _arg_topk = np.argsort(_scores)[-topk:]  # 从中选出新的topk
                _yid = [_yid[k] for k in _arg_topk]
                _scores = [_scores[k] for k in _arg_topk]
            yid = []
            scores = []
            for k in range(len(xid)):
                if _yid[k][-1] == 3:  # 找到<end>就返回
                    return id2str(_yid[k])
                else:
                    yid.append(_yid[k])
                    scores.append(_scores[k])
            yid = np.array(yid)
        # 如果50字都找不到<end>，直接返回
        return id2str(yid[np.argmax(scores)])

    s1 = u'核心观点：受营改增政策影响，公司16年销售收入同比下降2.75%（扣除政策影响，同比上升0.51%），资产减值损失的上升（同比增加1.08亿元）使得公司全年净利润同比下降75.28%，扣非后净利润亏损1911万元。若扣除公司对桂圳公司部分资产计提减值准备（减少净利润6476万元）与政府补助（2872万元）的影响，16年公司主业盈利约4400万元。16年公司主营业务毛利率同比上升2.92个百分点，由于收入的下降，期间费用率同比上升1.11个百分点，其中管理费用率同比上升2.91个百分点。16年公司投资收益同比增长23%，收到政府补贴合计2872万元，经营质量继续改善，经营活动现金流量净额同比提升10.16%。受上半年雨季时间较长的影响，16年公司景区共接待游客343.61万人次，同比增长5.48%，增速较15年有所放缓，其中漓江游船游客接待量同比下滑10.74%，漓江大瀑布饭店游客接待量同比增长8.65%，银子岩景区继续向好，游客接待量同比上升14.16%，收入与盈利分别同比提升11.19%与17.99%，两江四湖景区游客接待量同比上升5.82%，收入与盈利分别同比提升7.62%与253.72%（政府补助影响），收入、利润再创历史新高，两江四湖景区于2017年2月27日成功晋升国家5A级景区。海航完成入股强化公司体制改善与外延扩张预期，进一步提升公司市值向上弹性。作为桂林市旅游业唯一上市公司平台，桂林市政府通过公司平台整合相关资产的战略方向明确，随着海航成为公司第二大股东（16年底完成过户），后续海航将逐步加大对上市公司的影响力以及资源的导入，公司也有望进一步引入市场化激励机制，经营层面放开思路，投资层面提升效率与回报率，加快整合区域优质资源。后续公司资产中的亏损景区（其中丰鱼岩股权拟挂牌出售）将加快盘活或剥离处理，进一步提升主业盈利水平。公司在年报中提及围绕做大市值目标，争取3-5年内实现不少于一次再融资，通过外延扩张加快在行业内与关联行业的并购力度与速度。由桂林当地丰富的旅游资源与深厚的文化积淀入手，国资、海航与管理层共同推动公司后续的资本运作值得期待。财务预测与投资建议：根据年报，我们略上调公司未来3年投资收益与营业外收入，预计公司2017-2019年每股收益分别为0.19元、0.24元与0.29元（原预测2017-2018年每股收益为0.20元与0.26元），维持公司2017年10倍PS，对应市值55亿元，目标价15.29元，维持公司“买入”评级。风险提示：游客人次增长、国企改革与资本运作进展低于预期等。'
    s2 = u'事件：2月23日中国铝业发布公告，公司将于2月26日复牌，停牌期间公司通过市场化债转股的形式融资127亿元用来降低资产负债率，改善企业的资本结构，公司预计这将节约7亿元的财务费用。投资要点：增资改善公司资本结构，降低利息费用。中国铝业通过市场化债转股的方式引入包括华融瑞通、中国人寿、招平投资、中国信达、太保寿险、中银金融、工银金融、农银金融共8家机构，对子公司包头铝业、中铝矿业、中铝山东和中铝中州进行增资。目前中国铝业计划以6元/股的价格发行211,728.08万股以募集1,270,368.46万元，根据该等债务对应的利率情况，该等债务的一年利息费用合计约为7亿元，实施本次市场化债转股后，标的公司将相应减少利息费用。公司电解铝产能继续增加。2017年中铝旗下共有4家电解铝企业通电投产，分别为广西华磊新材料有限公司40万吨电解铝项目、山西中铝华润有限公司50万吨电解铝项目、贵州华仁铝业50万吨电解铝项目，内蒙古华云新材料有限公司78万吨电解铝项目，总计218万吨新增电解铝产能，公司拥有权益产能95万吨，公司电解铝合规产能大幅增加。利空消息基本出尽，铝价接近底部。2018年一季度各种利空消息打压铝价，其中包括：一季度国内迎来消费淡季、电解铝库存高企、采暖季关停产能3月将迎来复产、阿拉丁预计一季度可能有140万吨电解铝产能通电投产。但截至年前2月8日，受制于电解铝行业的低利润水平，据百川资讯统计，一季度投产电解铝产能仅22.42万吨。此外，我们预计采暖季结束后，成本曲线末端的限产产能没有动力复产，但需求将迎来复苏，电解铝库存有望迎来去化。综上所述，利空消息基本出尽，铝价接近底部，消费旺季叠加需求复产，二季度铝价有望冲向15000元/吨。2018-2019年铝价有望逐步回升，走出慢牛行情。按照现在成本端的情况来看，全年如果不出现大的政策性变化（比如继续关停产能、环保限产等），2018年二季度后铝价的运行区间大概率会在14500-15500元/吨，二季度可能冲向15000元/吨，之后会在14500-15500元/吨之间震荡。二季度短期内可能会形成电解铝需求大于运行产能的情况，但考虑到高库存，电解铝较强的供给弹性以及待投产产能处于成本曲线相对靠前的位置，2018年铝价无法形成持续上涨的趋势。但2019年随着所有合规产能逐渐被需求的增长消化，铝价有望继续沿着供给的成本曲线爬升至16000-16500元/吨的区间。盈利预测和投资评级。暂不考虑本次发行股份摊薄因素的影响，预计公司2017-2019的EPS分别为0.09元、0.14元、0.35元，对应当前股价的PE分别为88.66倍、59.28倍、23.31倍。由于公司在高点停牌，复牌后将面临补跌。我们下调公司A股的评级到增持评级。风险提示：电解铝下游需求受宏观经济影响不及预期；铝价下跌的风险；烧碱、煤沥青、石油焦等氧化铝和预焙阳极的原材料价格大幅上涨；公司电解铝、氧化铝等产品产量不及预期的风险。'
    s3 = u'公司近期召开年度经销商大会并披露集团和股份经营数据，超量完成全年任务，展望明年更加积极。从16年披露数据看，我们认为一是收入利润有保留；二是利润增速慢于收入增速不完全是消费税因素，预计明年利润率略有回升。我们保守预测17-18年EPS16.03、19.52元，按真实发货业绩继续给予1年目标价400元，继续强烈推荐。收入增长略超市场预期，利润增长略低市场预期，我们判断公司业绩有大量保留。16年收入增速19.2%，Q4收入增26.9%，超出市场预期的15%，不过我们认为其仍有保留，原因在于集团9-12月披露收入增速仅3%，大大小于1-8月的33%，而这明显不符合4季度需求加速爆发的事实，如果延续1-8月增速集团收入应有550多亿；16年净利润增长7.4%，低于市场预期的全年10%；其中Q4营业利润率47.2%，净利率34.0%，分别同比下降13.2和8个pct，数据异常。我们认为除消费税有较多增长外，费用也有大量确认，尽管系列酒需要增加大量投入，但不足以支出如此巨量，详见正文拆解。17年计划投放2.6万吨，普通飞天供应更紧张，批价可能再涨。公司在经销商会上明确提出，17年茅台酒的市场投放量为2.6万吨，相比16年增加3千吨，同比增长13%，但增量部分主要用于新品、生肖酒和国际市场，其他一律不增量，我们认为生肖酒多为礼品及收藏需求，开瓶率不高，国际市场不计入国内经销商份额，国内茅台流通量增长不多。我们认为公司来年不会上调出厂价，但会通过计划外价格及个性化酒提升均价。公司13年茅台基酒产量3.85万吨，按照4年后80%左右比例（考虑挥发及存于基酒），17年可供销量3.08万吨。但从今年实际发货看（可能接近3万吨），来年需求稍微有所增长，供应就会更为紧张，“茅台明年会更好”。我们判断未来三年茅台供需紧张加剧，带动批价被动上行，18-19年甚至可能重现历史高点。营销措施更接地气，全力打造世界顶级品牌。值得关注的是，公司本次首次提出，要增强对经销商服务意识，做好物流配送等服务工作，保证春节不断货，同时重视消费者体验，稳定产品价格，满足个性化需求等。开展“茅二代”、仪狄巨匠金奖、忠诚茅粉、茅台智库等形式，进一步集聚优质资源。目前三大名酒中，唯有茅台，在经销商和消费者中都具有一致赞誉，公司的良性经营在循环加速。营销措施更接地气，始终尊重渠道和消费者，并伴随一带一路开拓海外，全力打造世界顶级品牌。上调未来两年复合增速到20%，维持1年5000亿市值目标。公司年度经销商大会明确表示今年实现了经销商利润回归，确立了新的卖方市场供需关系，公司已经步入了上升发展新周期，走上了更加良性的发展轨道，对明年更有信心。集团目标收入增长20%，已经指明未来业绩增速。我们略上调17-18年EPS16.03、19.52元，按真实业绩估值，我们仍给予1年目标价400元，坚定强烈推荐。风险提示：基酒不足，淡季需求疲软'


    class Evaluate(Callback):
        def __init__(self):
            self.lowest = 1e10

        def on_epoch_end(self, epoch, logs=None):
            # 训练过程中观察一两个例子，显示标题质量提高的过程
            print(gen_title(s1))
            print(gen_title(s2))
            print(gen_title(s3))
            # 保存最优结果
            if logs['loss'] <= self.lowest:
                self.lowest = logs['loss']
                model.save_weights('s2s_summer_weights.h5')
                model.save('s2s_summer_modle.h5')

    #model = SimpleSeq2Seq(input_dim=1500, hidden_dim=10, output_length=130, output_dim=101)
    model = SimpleSeq2Seq(input_dim=1500, hidden_dim=10, output_length=130, output_dim=1500)
    model.compile(loss='mse', optimizer='rmsprop')

    evaluator = Evaluate()
    steps_per_epoch = math.ceil(len(lines) / batch_size)
#    model.fit_generator(data_generator(),
#                    steps_per_epoch=steps_per_epoch,
#                    epochs=epochs,
#                    callbacks=[evaluator])

    X = []
    Y = []
    for line in lines:
        line_list = line.split('\t')
        if 2 != len(line_list):
            continue
        title = line_list[0]
        content = line_list[1]

        X.append(str2id(content))
        Y.append(str2id(title, start_end=True))

    X = np.array([padding(X)])
    Y = np.array([padding(Y)])

    print(X)
    print(len(X))
    print(X.shape)
    print(Y)
    print(len(Y))
    print(Y.shape)

    model.fit(X, Y, batch_size=10, epochs=20)

    x_test = []
    x_test.append(str2id(s1))
    x_test = np.array([padding(x_test)])

    result = model.predict(x_test)
    print(result)
    print(result.shape)

    print("=========keras_lstm end==========")


