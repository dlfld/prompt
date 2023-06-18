"""
    数据处理过程当中公共的工具类
"""
from typing import List

import logddd
import torch
import sys

sys.path.append("..")
from model_params import Config


def data_reader(filename: str) -> List[str]:
    """
        读取文件，以行为单位返回文件内容
    :param filename: 文件路径
    :return: 以行为单位的内容list
    """
    with open(filename, "r") as f:
        return f.readlines()


def batchify_list(data, batch_size):
    """
        将输入数据按照batch_size进行划分
        @param data:输入数据
        @param batch_size: batch_size
        @return batch_data
    """
    batched_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batched_data.append(batch)

    return batched_data


def calcu_loss(total_scores, batch, loss_func_cross_entropy):
    """
        计算loss
        @param total_scores： 一个batch计算过程中的score矩阵
        @param 一个batch的数据
        @return loss
    """
    # =============================

    total_loss = 0
    for index, item in enumerate(batch):
        labels = item["labels"]
        onehot_labels = []
        # 依次获取所有的label
        for label_idx in range(len(labels)):
            item = labels[label_idx]
            label = [x - 1 for x in item if x != -100]
            onehot_label = torch.eye(Config.class_nums)[label]
            onehot_labels.append(onehot_label.tolist())

        onehot_labels = torch.tensor(onehot_labels).to(Config.device)
        onehot_labels = torch.squeeze(onehot_labels, dim=1)
        cur_scores = torch.tensor(total_scores[index], requires_grad=True).to(Config.device)
        cur_loss = loss_func_cross_entropy(cur_scores, onehot_labels)
        total_loss += cur_loss
    #     del cur_loss, onehot_labels
    # del total_scores
    return total_loss / Config.batch_size
