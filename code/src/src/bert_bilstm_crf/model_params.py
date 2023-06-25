class Config(object):
    """
        配置类，保存配置文件
    """
    dataset = "ctb"
    # 训练集位置
    # train_dataset_path = "/home/dlf/prompt/code/data/jw/short_data_train.txt"
    # train_dataset_path = "/home/dlf/prompt/code/data/jw/pos_seg.txt"
    # train_dataset_path = "/home/dlf/prompt/code/data/jw/mini_data.txt"
    train_dataset_path = "/home/dlf/prompt/code/data/jw/mini_data.txt"
    # 测试集位置
    test_dataset_path = "/home/dlf/prompt/code/data/jw/short_data_test.txt"
    # test_dataset_path = "/home/dlf/prompt/code/data/jw/mini_data.txt"
    dataset_path = "/home/dlf/prompt/code/data/jw/after_pos_seg.txt"
    # dataset_path = "/home/dlf/prompt/code/data/jw/PeopleDaily199801.txt"
    # train_dataset_path = "/home/dlf/prompt/dataset.csv"
    # 预训练模型的位置
    # bert
    # model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
    # medbert
    # model_checkpoint = "/home/dlf/prompt/code/model/medbert"
    # bart
    model_checkpoint = "/home/dlf/prompt/code/model/bart-large"
    # batch_size
    batch_size = 1
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
    # class_nums = 18
    # class_nums = 46 if dataset == "ctb" else 18
    class_nums = 42
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
    train_1_9_path = "/home/dlf/prompt/code/data/ctb/split_data/1_9_split/train_{idx}.data"
    test_1_9_path = "/home/dlf/prompt/code/data/ctb/split_data/1_9_split/test_{idx}.data"

    # 是否断点续训
    resume = True
    # few-shot 划分的数量
    few_shot = [5, 10, 15, 20, 25, 50, 75, 100, 200, 500]
    # few_shot = [50, 70]

    # 测试集位置
    test_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/ctb_test.data"
    # train dataset template
    train_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/fold/{item}.data"
    # label
    special_labels = ["[PLB]", "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
                      "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
                      "VA", "VE",
                      "NT-SHORT", "AS-1", "PN", "MSP-2", "NR-SHORT", "DER",
                      "URL", "DEC", "FW", "IJ", "NN-SHORT", "BA", "NT", "MSP", "LB",
                      "P", "NOI", "VV-2", "ON", "SB", "CS", "ETC", "DT", "AS", "M", "X",
                      "DEV"
                      ]
    pretrain_models = [
        "/home/dlf/prompt/code/model/bert_large_chinese",
        "/home/dlf/prompt/code/model/medbert",
        "/home/dlf/prompt/code/model/bart-large"]
