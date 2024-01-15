import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2)
# plt.figure(figsize=(40,40))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# 调整子图间距
plt.subplots_adjust(left=None, bottom=0, right=None, top=0.80, wspace=None, hspace=0.25)
# plt.figure(figsize=(80, 40))
plt.tight_layout()  # 调整图像布局
#  设置图例位置


# 画第1个图：中医临床切诊描述数据集
x = ['5', '10', '15', '20', '25']
crf = [0.513, 0.622, 0.690, 0.707, 0.736]
bilstm_crf = [0.429, 0.442, 0.408, 0.449, 0.444]
prompt = [0.612, 0.754, 0.798, 0.821, 0.840]
link = [0.645, 0.784, 0.825, 0.835, 0.855]

ax[0][0].plot(x, crf, label="PLM+CRF", c="blue", markerfacecolor='blue', markersize=6, marker="o")
ax[0][0].plot(x, bilstm_crf, label="PLM+BiLSTM+CRF", c="green", markersize=6, marker="v")
# ax[0][0].plot(x, prompt, label="PLM+Prompt+HMM", c="red", markersize=6, marker="p")
ax[0][0].plot(x, link, label="PLM+Prompt+Link", markersize=6, marker="*")

ax[0][0].set_title("中医临床切诊描述数据集", fontproperties='SimHei', fontsize=10)
front_size = 9
# 写数值在图的每一个结点上
# for a, b in zip(x, p-tuning):
#     ax[0][0].text(a, b, b, ha='center', va='bottom', fontsize=front_size)
# for a, b in zip(x, bilstm_crf):
#     ax[0][0].text(a, b, b, ha='center', va='bottom', fontsize=front_size)
# for a, b in zip(x, prompt):
#     ax[0][0].text(a, b, b, ha='center', va='bottom', fontsize=front_size)

# 设置图例位置
fig.legend(bbox_to_anchor=(0.85, 0.35))
#  CTB8.0数据集
x = ['5', '10', '15', '20', '25']
crf = [0.376, 0.373, 0.476, 0.474, 0.550]
bilstm_crf = [0.151, 0.187, 0.174, 0.171, 0.150]
prompt = [0.587, 0.666, 0.713, 0.739, 0.753]
link = [0.664, 0.706, 0.770, 0.788, 0.805]

ax[0][1].plot(x, crf, label="PLM+CRF", c="blue", markerfacecolor='blue', markersize=6, marker="o")
ax[0][1].plot(x, bilstm_crf, label="PLM+BiLSTM+CRF", c="green", markersize=6, marker="v")
# ax[0][1].plot(x, prompt, label="PLM+本文方法", c="red", markersize=6, marker="p")
ax[0][1].plot(x, link, label="PLM+Link", markersize=6, marker="*")

ax[0][1].set_title("CTB8.0数据集", fontproperties='SimHei', fontsize=10)
front_size = 9
# for a, b in zip(x, p-tuning):
#     ax[0][1].text(a, b, b, ha='center', va='bottom', fontsize=front_size)
# for a, b in zip(x, bilstm_crf):
#     ax[0][1].text(a, b, b, ha='center', va='bottom', fontsize=front_size)
# for a, b in zip(x, prompt):
#     ax[0][1].text(a, b, b, ha='center', va='bottom', fontsize=front_size)

# UD中文数据集
x = ['5', '10', '15', '20', '25']
crf = [0.339, 0.514, 0.553, 0.588, 0.612]
bilstm_crf = [0.234, 0.219, 0.228, 0.247, 0.267]
# prompt = [0.587, 0.666, 0.713, 0.739, 0.753]
link = [0.664, 0.706, 0.770, 0.788, 0.805]

ax[1][0].plot(x, crf, label="PLM+CRF", c="blue", markerfacecolor='blue', markersize=6, marker="o")
ax[1][0].plot(x, bilstm_crf, label="PLM+BiLSTM-CRF", c="green", markersize=6, marker="v")
# ax[1][0].plot(x, prompt, label="PLM+HMM", c="red", markersize=6, marker="p")
ax[1][0].plot(x, link, label="PLM+Link", markersize=6, marker="*")
ax[1][0].set_title("U D中文数据集", fontproperties='SimHei', fontsize=10)
front_size = 9
# for a, b in zip(x, p-tuning):
#     ax[1][0].text(a, b, b, ha='center', va='bottom', fontsize=front_size)
# for a, b in zip(x, bilstm_crf):
#     ax[1][0].text(a, b, b, ha='center', va='bottom', fontsize=front_size)
# for a, b in zip(x, prompt):
#     ax[1][0].text(a, b, b, ha='center', va='bottom', fontsize=front_size)
# for a, b in zip(x, link):
#     ax[1][0].text(a, b, b, ha='center', va='bottom', fontsize=front_size)

ax[1][1].remove()
plt.show()
