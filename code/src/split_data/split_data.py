"""
    当前程序任务是：划分训练数据和测试数据并把他们存下来
        1. 先将数据三七分 3为训练数据 7为测试数据  保存一份
        2. 将3分的数据分别划分为5 10 （15） 20 25
        3. 每一组划分为5组并分别存起来

"""

import sys
from sklearn.model_selection import train_test_split
import joblib

sys.path.append("..")
from data_process.data_processing import load_data


def split_train_test():
    """
        将原始数据三七分，并保存起来
    """
    # 加载标准数据
    standard_data = load_data("/home/dlf/prompt/code/data/jw/after_pos_seg.txt")
    y = [0] * len(standard_data)
    train, test, _, _ = train_test_split(standard_data, y, test_size=0.7, random_state=42)
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
        total_data.append(data)
    joblib.dump(total_data, save_path)


if __name__ == '__main__':
    data_list = [5, 10, 15, 20, 25]
    for index, item in enumerate(data_list):
        split_data_train(5, item, f"/home/dlf/prompt/code/data/split_data/{item}/{item}.data")
