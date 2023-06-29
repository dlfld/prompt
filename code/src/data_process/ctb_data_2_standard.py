"""
    将ctb数据转换成标准数据集

"""
from typing import List, Any, Dict

import logddd


def format_data_type_ctb(datas: List[str]) -> List[List[str]]:
    """
       对ctb数据集进行标准化处理
       中国	NR
       经济	NN
       简讯	NN
       to 中国/经济/简讯，‘NR/NN/NN’
    """

    # 所有数据
    res = []
    # 当前句子暂存
    temp_data = [
        [], []
    ]
    total_labels = set([])
    total_word = 0
    for data in datas:
        # print(data)
        data = data.replace("\n", "").strip()
        if len(data) == 0:
            res.append(["/".join(temp_data[0]), "/".join(temp_data[1])])
            temp_data = [
                [], []
            ]
        else:
            item = data.split("\t")
            # print(item)
            temp_data[0].append(item[0])
            temp_data[1].append(item[1])
            total_labels.add(item[1])
            total_word += 1

    for item in total_labels:
        print(item)
    logddd.log("数据总量为：", len(res))
    logddd.log("词的总量为：", total_word)
    return res


def get_all_ctb_data():
    with open("/home/dlf/prompt/code/data/ctb/totaldata.txt", "r") as f:
        return f.readlines()


if __name__ == '__main__':
    all_data = get_all_ctb_data()
    format_data_type_ctb(all_data)
