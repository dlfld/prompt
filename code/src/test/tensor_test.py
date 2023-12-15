import logddd
import torch
import numpy as np

# 创建一个示例数组
arr = np.array([[1, 2, 3],
                [8, 7, 6],
                [7, 8, 9]])

# 沿着每个维度找到最大值的索引
max_indices_axis_0 = np.argmax(arr, axis=0)  # 沿着第一个轴找到最大值的索引
max_indices_axis_1 = np.argmax(arr, axis=1)  # 沿着第二个轴找到最大值的索引

print("每个维度最大值的索引（沿着轴0）：", max_indices_axis_0)
print("每个维度最大值的索引（沿着轴1）：", max_indices_axis_1)
