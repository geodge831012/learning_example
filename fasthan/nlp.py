from fastHan import FastHan

if __name__ == '__main__':

    #model = FastHan(model_type='base')
    model = FastHan(model_type='large')

    #model.set_device('cuda:0')
    #model.set_device('cpu')

    #sentence = ["我爱踢足球", "林丹是冠军", "陆股通减持个股中，最新持股比例较前一日减少0.3%的共有27只，减少比例最多的是天宇股份，陆股通最新持股量为127.52万股，占流通股比例为2.36%，持股占比较上一日减少1.71%；其次是长城科技，持股比例较上一日减少1.51%；持股比例减少较多的还有上海梅林等。", "精心筹备春季广交会，帮助企业深化国际合作、拓展国际市场。有力促进国内经济发展，维护全球供应链稳定。胡春华、肖捷、何立峰陪同考察。", "他强调，要贯彻习近平总书记重要讲话精神，按照党中央、国务院决策部署，统筹推进疫情防控和经济社会发展，推动商贸物流市场正常运行，更大力度深化改革开放，释放巨大消费潜力，稳住外贸外资基本盘，增强经济发展动力。"]

    # 句子字符长度不能超过512字节
    sentence = '迈向充满希望的新世纪——一九九八年新年讲话(附图片1张),中共中央总书记、国家主席江泽民,(一九九七年十二月三十一日),12月31日,中共中央总书记、国家主席江泽民发表1998年新年讲话《迈向充满希望的新世纪》。(新华社记者兰红光摄),同胞们、朋友们、女士们、先生们:'
    #sentence = 'abc'

    #answer = model(sentence, 'Parsing')
    #answer = model(sentence, 'CWS')
    #answer = model(sentence, 'POS')
    answer = model(sentence, 'NER')

    print(type(answer))
    print(type(answer[0]))
    print(type(answer[0][0]))
    print(answer)

    for i, sentence in enumerate(answer):
        print(i)
        #print(type(sentence))
        #print(sentence)
        for token in sentence:
            #print(type(token))
            #print(token)

            # pos、head、head_label、ner
            # 词性、依存关系、命名实体识别信息
            #print(token, token.pos, token.head, token.head_label, token.ner)
            print(type(token))
            print(type(token.ner))
            print(str(token))
            print(token, token.ner)
            print(type(token.word))
            print(token.word)
