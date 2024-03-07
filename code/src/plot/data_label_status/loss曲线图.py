import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
# 处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置字体为宋体
plt.rcParams['font.size'] = 12  # 设置字号为12
# 给定的Loss数据
loss_data = [
]
with open("mcbert-crf.txt", "r") as f:
    datas = f.readlines()
loss_data = [float(x.replace("/n", "")) for x in datas]
loss_data = loss_data[:50]
print(loss_data)
# 创建图表
plt.figure(figsize=(10, 5))
# 绘制原始的Loss曲线，不显示数据点
plt.plot(loss_data, linestyle='-', color='blue', label='BAP+CH')
# plt.xticks(np.arange(len(loss_data)))
# 设置标题和标签
# plt.title('Original Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# 显示图表
plt.grid(True)
plt.show()
