"""
    当前程序任务是：划分训练数据和测试数据并把他们存下来
        1. 先将数据三七分 3为训练数据 7为测试数据  保存一份
        2. 将3分的数据分别划分为5 10 （15） 20 25
        3. 每一组划分为5组并分别存起来

"""

import sys

import joblib
import logddd
from sklearn.model_selection import train_test_split, KFold

sys.path.append("..")
from data_process.data_processing import load_data, load_ctb_data
from data_process.data_processing import load_ud_en_data


def split_data_train(data_num, sampling_nums, save_path):
    """
        按照输入的采样数和采样轮次进行采样
        @param data_num:采样轮次
        @param sampling_nums: 采样数
    """
    test_data = joblib.load("/home/dlf/crf/code/data/split_data/pos_seg_train.data")
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
    data_list = [5, 10, 15, 20, 25, 50, 75, 100, 200, 500]
    # data_list = [50, 75]
    for index, item in enumerate(data_list):
        split_data_train(5, item, f"/home/dlf/crf/code/data/split_data/{item}/{item}.data")


def split_dataset_1_9():
    """
     一九分数据集
    """
    save_path = "/home/dlf/crf/code/data/split_data/1_9_split"
    standard_data = load_data("/home/dlf/crf/code/data/jw/after_pos_seg.txt")
    y_none_use = [0] * len(standard_data)
    kfold = KFold(n_splits=10, shuffle=True)
    item = 1
    for train, val in kfold.split(standard_data, y_none_use):
        test_standard_data = [standard_data[x] for x in train]
        train_standard_data = [standard_data[x] for x in val]
        logddd.log(len(test_standard_data), len(train_standard_data))
        joblib.dump(train_standard_data, f"{save_path}/train_{item}.data")
        joblib.dump(test_standard_data, f"{save_path}/test_{item}.data")
        item += 1


def save(path, datas):
    with open(path, "w") as f:
        f.writelines(datas)


def split_ctb_dataset_1_9():
    """
     一九分数据集
    """
    save_path = "/home/dlf/crf/code/data/ctb/split_data/1_9_split"
    standard_data = load_ctb_data()
    y_none_use = [0] * len(standard_data)
    kfold = KFold(n_splits=10, shuffle=True)
    item = 1
    for train, val in kfold.split(standard_data, y_none_use):
        test_standard_data = [standard_data[x] for x in train]
        train_standard_data = [standard_data[x] for x in val]
        logddd.log(len(test_standard_data), len(train_standard_data))
        joblib.dump(train_standard_data, f"{save_path}/train_{item}.data")
        joblib.dump(test_standard_data, f"{save_path}/test_{item}.data")
        item += 1


def split_train_test(test_size=0.7):
    """
        将原始数据三七分，并保存起来
    """
    # 加载标准数据
    # standard_data = load_data("/home/dlf/crf/code/data/jw/after_pos_seg.txt")
    standard_data = load_data("/home/dlf/crf/code/data/jw/after_pos_seg.txt")
    y = [0] * len(standard_data)
    train, test, _, _ = train_test_split(standard_data, y, test_size=test_size, random_state=42)
    joblib.dump(test, "ctb_test.data")
    joblib.dump(train, "ctb_train.data")


def split_data_few_shot(save_path, datas, data_split, fold):
    """
    传进来的是3的数据，然后将3的数据划分成5份，从5份里面去取
    """

    kfold = KFold(n_splits=fold, shuffle=True, random_state=66)
    # 将数据划分为fold份
    total_split = [val for _, val in kfold.split(datas, [0] * len(datas))]
    # 遍历所有数据条数
    for item in data_split:
        total_datas = []
        total_ids = []
        # 遍历每一份数据
        for val in total_split:
            fold_datas = []
            # 从每一份数据中取出item条
            selected = random.sample(val.tolist(), item)
            for select_id in selected:
                fold_datas.append(datas[select_id])
            total_datas.append(fold_datas)
            total_ids.append([selected])

        logddd.log(len(total_datas))
        joblib.dump(total_datas, f"{save_path}/{item}.data")
        joblib.dump(total_ids, f"{save_path}/{item}_ids.data")


def split_data(test_size, datas):
    """
        划分数据集
    """
    train, test = train_test_split(datas, test_size=test_size, random_state=42)
    train_index = [str(datas.index(item)) + "\n" for item in train]
    test_index = [str(datas.index(item)) + "\n" for item in test]
    # 保存数据的index
    save("train_index.txt", train_index)
    save("test_index.txt", test_index)
    joblib.dump(test, "ctb_test.data")
    joblib.dump(train, "ctb_train.data")


import random


def extract_items(lst, num_items):
    """
     从指定列表中抽取出指定条数的数据
    """
    if num_items > len(lst):
        raise ValueError("The number of items to extract cannot be greater than the list length.")

    extracted_items = random.sample(lst, num_items)
    print(extracted_items)
    return extracted_items


def split_ud():
    """
    因为数据集已经划分好了训练集测试集和验证集
    那么我们就在训练集里面采样，首先，将数据集划分为5份，然后采样5-500
    """
    # 加载数据
    data_path = "/Users/dailinfeng/Desktop/prompt/code/data/ud/ud_en/zh_gsdsimp-ud-train.conllu"
    datas = load_ud_en_data(data_path)
    # joblib.dump(datas,"/Users/dailinfeng/Desktop/crf/code/data/ud/ud_en/train.data")
    # logddd.log(len(datas))
    split_data_few_shot("/Users/dailinfeng/Desktop/prompt/code/data/ud/ud_en/fold/", datas,
                        [5, 10, 15, 20, 25, 50, 75, 100, 200, 500], 5)


if __name__ == '__main__':
    # =================================================将数据三七分==========================================
    # datas = load_data("/home/dlf/crf/code/data/jw/after_pos_seg.txt")
    # split_data(test_size=0.7, datas=datas)
    # datas = load_ctb_data()

    split_ud()
    # split_data(test_size=0.7, datas=datas)
    # =================================================将数据三七分==========================================

    # =================================================在三分的数据中划分数据==================================

    # datas = joblib.load("/home/dlf/crf/code/data/split_data/pos_seg_train.data")
    # split_data_few_shot("/home/dlf/crf/code/data/ctb/split_data/few_shot/fold/", datas,
    #                     [5, 10, 15, 20, 25, 50, 75, 100, 200, 500, 1000], 5)
    # =================================================在三分的数据中划分数据==================================
    # =================================================在七分的数据中抽取出指定条数的数据==========================
    # test_datas = joblib.load("/home/dlf/crf/code/data/ctb/split_data/few_shot/ctb_test.data")
    # one_tentn_test_datas = extract_items(test_datas, 1500)
    # # 7137
    # ids = []
    # data_map = {}
    # for index, data in enumerate(datas):
    #     data_map[str(data)] = index
    # for data in one_tentn_test_datas:
    #     ids.append(data_map[str(data)])
    # joblib.dump(one_tentn_test_datas, "/home/dlf/crf/code/data/ctb/split_data/few_shot/test_1500.data")
    # ids_index = [str(item) + "\n" for item in ids]
    # save("/home/dlf/crf/code/data/ctb/split_data/few_shot/test_1500.txt", ids_index)

    # =================================================在七分的数据中抽取出指定条数的数据==========================

    pass
