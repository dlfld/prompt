class Config(object):
    """
        配置类，保存配置文件
    """
    dataset = "ud"

    # batch_size
    batch_size = 8
    # 学习率
    learning_rate = 2e-5
    # epoch数
    num_train_epochs = 100
    # 句子的最大补齐长度
    # sentence_max_len = 2048
    sentence_max_len = 128
    # 结果文件存储位置
    predict_res_file = "/home/dlf/prompt/code/res_files/short_data_res_{}.txt"
    # 词性的类别数量
    # jw
    # class_nums = 18
    # ctb
    # class_nums = 42
    # ud
    class_nums = 15
    # 计算使用的device
    # device = "cpu"
    device = "cuda:0"
    # k折交叉验证
    kfold = 1


    # ctb dataset
    # train_1_9_path = "/home/dlf/prompt/code/data/ctb/split_data/1_9_split/train_{idx}.data"
    # test_1_9_path = "/home/dlf/prompt/code/data/ctb/split_data/1_9_split/test_{idx}.data"

    # 是否断点续训
    resume = False
    # few-shot 划分的数量
    few_shot = [5, 10, 15, 20, 25, 50, 75, 100, 200, 500]
    # few_shot = [20, 70]

    # 测试集位置
    # test_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/ctb_test.data"
    # test_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/one_tentn_test_datas.data"
    # train dataset template
    # train_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/fold/{item}.data"
    
    test_data_path = "/home/dlf/prompt/code/data/ud/ud_en/test.data"
    train_data_path = "/home/dlf/prompt/code/data/ud/ud_en/fold/{item}.data"
    # log dir
    log_dir = "ud_bert_medbert_bart/"
    # label
    # ctb
    # special_labels = ["[PLB]", "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
    #                   "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
    #                   "VA", "VE",
    #                   "NT-SHORT", "AS-1", "PN", "MSP-2", "NR-SHORT", "DER",
    #                   "URL", "DEC", "FW", "IJ", "NN-SHORT", "BA", "NT", "MSP", "LB",
    #                   "P", "NOI", "VV-2", "ON", "SB", "CS", "ETC", "DT", "AS", "M", "X",
    #                   "DEV"
    #                   ]
    special_labels = ["[PLB]","PROPN", "SYM", "X", "PRON", "ADJ", "NOUN", "PART", "DET", "CCONJ", "ADP", "VERB", "NUM", "PUNCT", "AUX", "ADV"]
    pretrain_models = [
        "/home/dlf/prompt/code/model/bert_large_chinese",
        "/home/dlf/prompt/code/model/medbert",
        "/home/dlf/prompt/code/model/bart-large"
    ]
