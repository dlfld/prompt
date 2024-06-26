import matplotlib.pyplot as plt
plt.style.use('ggplot')
# 处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置字体为宋体
plt.rcParams['font.size'] = 12  # 设置字号为12
# 数据点
shots = [5, 10, 15, 20, 25]
f1_bert_chain_ptmm = [61.36, 73.60, 78.63, 81.13, 82.57]
f1_mc_bert_chain_ptmm = [64.07, 75.26, 81.11, 84.16, 84.65]
f1_bart_chain_ptmm = [64.51, 79.76, 82.52, 83.68, 85.52]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(shots, f1_bert_chain_ptmm, marker='o', linestyle='-', color='red', label='BERT+ours')
plt.plot(shots, f1_mc_bert_chain_ptmm, marker='s', linestyle='-', color='green', label='MC-BERT+ours')
plt.plot(shots, f1_bart_chain_ptmm, marker='^', linestyle='-', color='blue', label='BART+ours')

# 设置标题和标签
# plt.title('F1 Score Change Across Different Shots')
plt.xlabel('Shots')
plt.ylabel('F1')

# 设置x轴刻度
plt.xticks(shots)

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
