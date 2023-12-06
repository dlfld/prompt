"""
    蒋文数据集:pos_seg.txt 数据集的处理
    指定的标准数据集格式为  ['脉/数', 'NR/VA'] 。 以后的数据都处理成这个格式之后再往下处理
    当前数据集的格式为 脉 沉 细    NR VA VA    10138
"""
from typing import List, Any, Dict


def format_data_type_pos_seg(datas: List[str]) -> List[List[str]]:
    """
        对蒋文的数据集进行标准格式化处理 
        对数据格式进行更改，脉 数    NR VA    10592 -> ['脉/数', 'NR/VA']
    :param datas: eg: 脉 沉 细    NR VA VA    10138
    :return: 更改好的数据
    [
        ['脉/数', 'NR/VA'],
        ...
    ]
    """
    res = []
    for data in datas:
        # 脉 数    NR VA    10592 -> 脉 数 && NR VA && 10592
        data = data.replace("    ", "&&")
        # 脉 数&&NR VA&&10592 -> 脉/数&&NR/VA&&10592
        data = data.replace(" ", "/")
        # 脉/数&&NR/VA&&10592 -> ['脉/数', 'NR/VA']
        data = data.split("&&")[:-1]
        res.append(data)

    return res
