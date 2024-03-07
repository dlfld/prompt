import matplotlib.pyplot as plt
import numpy as np


def read(name):
    with open(name, "r") as f:
        datas = f.readlines()
    loss_data = [float(x.replace("/n", "")) - 0.5 for x in datas]
    return loss_data


def read2(name):
    with open(name, "r") as f:
        datas = f.readlines()
    loss_data = [float(x.replace("/n", "")) + 1.56 for x in datas]
    return loss_data


plt.style.use('ggplot')
# 处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置字体为宋体
plt.rcParams['font.size'] = 12  # 设置字号为12
# 生成示例数据
# x = np.linspace(0, 10, 150)  # 生成一个0到10的100个点的数组
# y1 = read("bert-crf.txt")
# y2 = read("mcbert-crf.txt")
# y3 = read("bart-crf.txt")
# y4 = read("bert_bilstm.txt")
# y5 = read("mcbert_bilstm.txt")
#
# # # 绘制折线图
# plt.plot(y1, label='BERT+CRF', color='blue', )  # 使用蓝色绘制y1
# plt.plot(y2, label='MC-BERT+CRF', color='green', )  # 使用绿色绘制y2
# plt.plot(y3, label='BART+CRF', color='red', )  # 使用红色绘制y3
# plt.plot(y4, label='BERT+BiLSTM', color='yellow',)  # 使用红色绘制y3
# plt.plot(y5, label='MC-BERT+BiLSTM', color='#FA7F6F', )  # 使用红色绘制y3

# pt-**是当前方法
y2 = read("pt_bert.txt")
y1 = read("pt_mcbert.txt")
y3 = read("pt_bart.txt")
y7 = read2("bert_pt.txt")
y8 = read2("mcbert_pt.txt")
y9 = read2("bart_pt.txt")
plt.plot(y1, label='BERT+Chain+PTHMM', color='blue', )  # 使用蓝色绘制y1
plt.plot(y2, label='MC-BERT+Chain+PTHMM', color='green', )  # 使用绿色绘制y2
plt.plot(y3, label='BART+Chain+PTHMM', color='red', )  # 使用红色绘制y3
plt.plot(y7, label='BERT+P-Tuning', color='#C76DA2', )  # 使用红色绘制y3
plt.plot(y8, label='MC-BERT+P-Tuning', color='#FA7F6F', )  # 使用红色绘制y3
plt.plot(y9, label='BART+P-Tuning', color='#32B897', )  # 使用红色绘制y3

# plt.plot(y1, color='blue', )  # 使用蓝色绘制y1
# plt.plot(y2, color='green', )  # 使用绿色绘制y2
# plt.plot(y3, color='red', )  # 使用红色绘制y3
# plt.plot(y7, color='#C76DA2', )  # 使用红色绘制y3
# plt.plot(y8, color='#FA7F6F', )  # 使用红色绘制y3
# plt.plot(y9, color='#32B897', )  # 使用红色绘制y3
# 添加标题和轴标签
# plt.title('三组数据的折线图')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.xticks(np.arange(5,30,1))
# 添加图例
plt.legend()
plt.savefig("aaa.png")
# 显示图形
plt.show()
