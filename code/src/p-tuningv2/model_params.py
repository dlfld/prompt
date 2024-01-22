class Config(object):
    """
        配置类，保存配置文件
    """
    # batch_size
    batch_size = 2
    # 连续提示块的数量
    prompt_length = 6
    prompt_encoder_type = "gru"
    # 学习率
    learning_rate = 2e-5
    hmm_lr = 0.01
    head_lr = 0.001
    pre_seq_len = 128
    # epoch数
    num_train_epochs = 100
    embed_size = 768

    hidden_dropout_prob = 0.2
    loss_file = "tuning_bert"

    prefix_hidden_size = 512
    # 句子的最大补齐长度
    # sentence_max_len = 2048
    sentence_max_len = 256

    # 结果文件存储位置
    predict_res_file = "/home/dlf/prompt/code/res_files/short_data_res_{}.txt"
    template6 = "在句子“{sentence}”中，词语“{word}”的前文如果是由“{pre_part_of_speech}”词性的词语“{pre_word}”来修饰，那么词语“{word}”的词性是“[MASK]”→ {part_of_speech}"
    template7 = "词语“{word}”的前文如果是由“{pre_part_of_speech}”词性的词语“{pre_word}”来修饰，在句子“{sentence}”中，那么词语“{word}”的词性是“[MASK]”→ {part_of_speech}"
    template8 = "在句子“{sentence}”中，词语“{word}”的词性是“[MASK]”,如果词语“{word}”的前文是由“{pre_part_of_speech}”词性的词语“{pre_word}”来修饰。→ {part_of_speech}"
    template9 = "在句子“{sentence}”中，词语“{pre_word}的词性是{pre_part_of_speech}”，那么词语“{word}”的词性是“[MASK]”→ {part_of_speech}"
    template_pt = "[T]{sentence}[T]{word}[T]{pre_part_of_speech}[T]{pre_word}[T]{word}[T]?[MASK]→ {part_of_speech}"
    template = template7
    # 词性的类别数量
    class_nums = 18
    # class_nums = 42
    # ud
    # class_nums = 15
    # 计算使用的device
    # device = "cpu"
    device = "cuda:0"
    # k折交叉验证
    kfold = 5
    # 1train9test 10折 train path
    # 1train9test 10折 test path
    # jw dataset
    # train_1_9_path = "/home/dlf/prompt/code/data/split_data/1_9_split/train_{idx}.data"
    # test_1_9_path = "/home/dlf/prompt/code/data/split_data/1_9_split/test_{idx}.data"

    # ctb dataset
    # train_1_9_path = "/home/dlf/prompt/code/data/ctb/split_data/1_9_split/train_{idx}.data"
    # test_1_9_path = "/home/dlf/prompt/code/data/ctb/split_data/1_9_split/test_{idx}.data"

    # 是否断点续训
    resume = False
    # few-shot 划分的数量
    few_shot = [5, 10, 15, 20, 25]
    # 测试集位置
    # jw
    train_data_path = "/home/dlf/prompt/code/data/split_data/fold/{item}.data"
    test_data_path = "/home/dlf/prompt/code/data/split_data/pos_seg_test.data"
    # CTB
    # test_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/one_tentn_test_datas.data"
    # train_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/fold/{item}.data"
    # test_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/test_3000.data"
    # UD
    # test_data_path = "/home/dlf/prompt/code/data/ud/ud_en/test.data"
    # train_data_path = "/home/dlf/prompt/code/data/ud/ud_en/fold/{item}.data"
    # log dir
    log_dir = "ud_bert_medbert_bert/"
    # train dataset template

    # train_data_path = "/home/dlf/prompt/code/data/split_data/{item}.data"
    # 截取句子的前n个词组成prompt,超过8个要oom
    # pre_n = 8
    # label
    # jw 数据集
    special_labels = ["[PLB]", "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
                      "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
                      "VA", "VE", "[T]"]
    # ctb数据集
    # special_labels = ["[PLB]", "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
    #                 "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
    #                "VA", "VE",
    #               "NT-SHORT", "AS-1", "PN", "MSP-2", "NR-SHORT", "DER",
    #              "URL", "DEC", "FW", "IJ", "NN-SHORT", "BA", "NT", "MSP", "LB",
    #             "P", "NOI", "VV-2", "ON", "SB", "CS", "ETC", "DT", "AS", "M", "X",
    #            "DEV"
    #           ]
    # UD 数据集
    # special_labels = ["[PLB]", "PROPN", "SYM", "X", "PRON", "ADJ", "NOUN", "PART", "DET", "CCONJ", "ADP", "VERB", "NUM",
    #                   "PUNCT", "AUX", "ADV"]
    # 检查点的保存位置
    checkpoint_file = "/home/dlf/prompt/code/src/prompt/pths/ud-ch_{filename}.pth"
    pretrain_models = [
        # "/home/dlf/prompt/code/model/bert_large_chinese",
        "/home/dlf/prompt/code/model/medbert",
        # "/home/dlf/prompt/code/model/bart-large"
    ]
