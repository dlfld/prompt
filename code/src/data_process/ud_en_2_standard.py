"""
    将UD_EN数据转换成标准数据集
"""
from typing import List


def get_all_ud_data(data_path):
    with open(data_path, "r") as f:
        return f.readlines()


def format_data_type_ud(datas: List[str]) -> List[List[str]]:
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
    # max_len = 0
    for data in datas:
        # print(data)
        data = data.replace("\n", "").strip()
        if len(data) == 0:
            res.append(["/".join(temp_data[0]), "/".join(temp_data[1])])
            temp_data = [
                [], []
            ]
        else:
            # 每一个句子的开头都有一个这个，如果是这个的话就直接跳过就是了
            if data.startswith("# sent_id") or data.startswith("# text"):
                continue
            item = data.split("\t")
            temp_data[0].append(item[1])
            temp_data[1].append(item[3])
            total_labels.add(item[3])
    # for item in total_labels:
    #     print(item)
    # print(len(total_labels))
    return res


if __name__ == '__main__':
    res = get_all_ud_data("/Users/dailinfeng/Desktop/prompt/code/data/ud/ud_en/zh_gsdsimp-ud-train.conllu")
    standard = format_data_type_ud(res)
