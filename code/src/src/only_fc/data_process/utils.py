"""
    数据处理过程当中公共的工具类
"""
from typing import List


def data_reader(filename: str) -> List[str]:
    """
        读取文件，以行为单位返回文件内容
    :param filename: 文件路径
    :return: 以行为单位的内容list
    """
    with open(filename, "r") as f:
        return f.readlines()
    
    
