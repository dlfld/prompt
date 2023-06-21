"""
    当前程序任务是：划分训练数据和测试数据并把他们存下来
        1. 先将数据三七分 3为训练数据 7为测试数据  保存一份
        2. 将3分的数据分别划分为5 10 （15） 20 25
        3. 每一组划分为5组并分别存起来

"""

import sys

import logddd
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib

sys.path.append("..")
from data_process.data_processing import load_data


def split_train_test(test_size=0.7):
    """
        将原始数据三七分，并保存起来
    """
    # 加载标准数据
    standard_data = load_data("/home/dlf/prompt/code/data/jw/after_pos_seg.txt")
    y = [0] * len(standard_data)
    train, test, _, _ = train_test_split(standard_data, y, test_size=test_size, random_state=42)
    joblib.dump(test, "pos_seg_test.data")
    joblib.dump(train, "pos_seg_train.data")


def split_data_train(data_num, sampling_nums, save_path):
    """
        按照输入的采样数和采样轮次进行采样
        @param data_num:采样轮次
        @param sampling_nums: 采样数
    """
    test_data = joblib.load("/home/dlf/prompt/code/data/split_data/pos_seg_train.data")
    total_data = []
    for i in range(data_num):
        data = test_data[i * sampling_nums + 1:i * sampling_nums + sampling_nums + 1]

        for item in data:
            # logddd.log(item)
            sequence = item[0].strip().split("/")
            # print(sequence)
        total_data.append(data)
    logddd.log(len(total_data))
    joblib.dump(total_data, save_path)


def split_dataset_3_7():
    """
        三七分的数据集
    """
    # data_list = [5, 10, 15, 20, 25]
    data_list = [50, 70]
    for index, item in enumerate(data_list):
        split_data_train(5, item, f"/home/dlf/prompt/code/data/split_data/{item}/{item}.data")


def split_dataset_1_9():
    """
     一九分数据集
    """
    save_path = "/home/dlf/prompt/code/data/split_data/1_9_split"
    standard_data = load_data("/home/dlf/prompt/code/data/jw/after_pos_seg.txt")
    y_none_use = [0] * len(standard_data)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    item = 1
    for train, val in kfold.split(standard_data, y_none_use):
        test_standard_data = [standard_data[x] for x in train]
        train_standard_data = [standard_data[x] for x in val]
        logddd.log(len(test_standard_data), len(train_standard_data))
        joblib.dump(train_standard_data, f"{save_path}/train_{item}.data")
        joblib.dump(test_standard_data, f"{save_path}/test_{item}.data")
        item += 1


def generate_data_for_low_resource_seq_labling():
    standard_data = load_data("/home/dlf/prompt/code/data/jw/after_pos_seg.txt")
    y = [0] * len(standard_data)
    train, test, _, _ = train_test_split(standard_data, y, test_size=0.9, random_state=42)
    train_datas = []
    test_datas = []
    for sequence in train:
        # print(sequence)
        # exit(0)
        for item in zip(sequence[0].split("/"), sequence[1].split("/")):
            train_datas.append(item[0] + " " + item[1] + "\n")
        train_datas.append("EOS EOS\n")
    for sequence in test:
        # print(sequence)
        # exit(0)
        for item in zip(sequence[0].split("/"), sequence[1].split("/")):
            test_datas.append(item[0] + " " + item[1] + "\n")
        test_datas.append("EOS EOS\n")
    save("train.txt", train_datas)
    save("test.txt", test_datas)


def save(path, datas):
    with open(path, "w") as f:
        f.writelines(datas)


if __name__ == '__main__':
    generate_data_for_low_resource_seq_labling()
