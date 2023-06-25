class Config(object):
    """
        配置类，保存配置文件
    """
    # 训练集位置
    # train_dataset_path = "/home/dlf/prompt/code/data/jw/short_data_train.txt"
    # train_dataset_path = "/home/dlf/prompt/code/data/jw/pos_seg.txt"
    # train_dataset_path = "/home/dlf/prompt/code/data/jw/mini_data.txt"
    # train_dataset_path = "/home/dlf/prompt/code/data/jw/mini_data.txt"
    # 测试集位置
    # test_dataset_path = "/home/dlf/prompt/code/data/jw/short_data_test.txt"
    # test_dataset_path = "/home/dlf/prompt/code/data/jw/mini_data.txt"
    # dataset_path = "/home/dlf/prompt/code/data/jw/after_pos_seg.txt"
    # dataset_path = "/home/dlf/prompt/code/data/jw/PeopleDaily199801.txt"
    # train_dataset_path = "/home/dlf/prompt/dataset.csv"
    # 预训练模型的位置
    # bert
    # model_checkpoint = "/home/kdwang/dlf/prompt/code/model/bert_large_chinese"
    # medbert
    # model_checkpoint = "/home/kdwang/dlf/prompt/code/model/medbert"
    # bart
    model_checkpoint = "/home/dlf/prompt/code/model/bart-large"
    # batch_size
    batch_size = 4
    # 学习率
    learning_rate = 2e-5
    # epoch数
    num_train_epochs = 94
    # 句子的最大补齐长度
    # sentence_max_len = 2048
    sentence_max_len = 128
    # 结果文件存储位置
    predict_res_file = "/home/dlf/prompt/code/res_files/short_data_res_{}.txt"
    # 词性的类别数量
    class_nums = 18
    # 计算使用的device
    # device = "cpu"
    device = "cuda:0"
    # k折交叉验证
    kfold = 10
    # few-shot 划分的数量
    few_shot = [5, 10, 15, 20, 25]
    # few_shot = [50, 70]
    # 1train9test 10折 train path
    train_1_9_path = "/home/dlf/prompt/code/data/split_data/1_9_split/train_{idx}.data"
    # 1train9test 10折 test path
    test_1_9_path = "/home/dlf/prompt/code/data/split_data/1_9_split/test_{idx}.data"
    # # 测试集位置
    # test_data_path = "/home/kdwang/dlf/prompt/code/data/split_data/ctb_test.data"
    # # train dataset template
    # train_data_path = "/home/kdwang/dlf/prompt/code/data/split_data/{item}/{item}.data"

    test_data_path = "/home/dlf/prompt/code/data/split_data/pos_seg_test.data"
    # train dataset template
    train_data_path = "/home/dlf/prompt/code/data/split_data/{item}/{item}.data"
    # 是否断点续训
    resume = False
