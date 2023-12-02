class Config(object):
    """
        配置类，保存配置文件
    """
    # 训练集位置
    train_dataset_path = "/home/dlf/crf/code/data/jw/short_data_train.txt"
    # train_dataset_path = "/home/dlf/crf/code/data/jw/pos_seg.txt"
    # train_dataset_path = "/home/dlf/crf/code/data/jw/short_data_test.txt"
    # 测试集位置
    test_dataset_path = "/home/dlf/crf/code/data/jw/short_data_test.txt"
    # crf dataset
    # train_dataset_path = "/home/dlf/crf/dataset.csv"
    # 预训练模型的位置
    model_checkpoint = "/home/dlf/crf/code/model/bert_large_chinese"
    # batch_size
    batch_size = 1
    # 学习率
    learning_rate = 2e-5
    # epoch数
    num_train_epochs = 3
    # 句子的最大补齐长度
    sentence_max_len = 128
    # 结果文件存储位置
    predict_res_file = "/home/dlf/crf/code/res_files/short_data_res_{}.txt"
    # 词性的类别数量
    class_nums = 18
    # 计算使用的device
    # device = "cuda:0"
    device = "cpu"

