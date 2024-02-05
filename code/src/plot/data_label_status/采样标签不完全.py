import matplotlib.pyplot as plt

# 设置绘图风格
plt.style.use('ggplot')
# 处理中文乱码
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置字体为宋体
plt.rcParams['font.size'] = 12  # 设置字号为12
total_tags = 18
dataset_sizes = [5, 10, 15, 20, 25]
tag_counts = [6.6, 8.4, 9.6, 10.8, 11.2]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(dataset_sizes, tag_counts, marker='o', linestyle='-', color='blue', label='标签平均覆盖数量')

# 添加总标签数的水平线
plt.axhline(y=total_tags, color='red', linestyle='--', label='总标签数(18)')

# 添加标题和标签
plt.title(u'训练集中的词性标签覆盖情况', fontsize=12)
plt.xlabel(u'训练数据中的句子数量', fontsize=12)
plt.ylabel(u'训练集中平均覆盖的标签数量', fontsize=12)
# plt.title(u'                ', fontsize=15)
# plt.xlabel(u'               ', fontsize=12)
# plt.ylabel(u'                 ', fontsize=12)
plt.xticks(dataset_sizes)
plt.legend()

# 在每个点上标注标签覆盖数量
for i, count in enumerate(tag_counts):
    plt.text(dataset_sizes[i], count+0.2, f'{count}', ha='center', color='blue', fontsize=10)

plt.grid(True)
plt.tight_layout()
plt.savefig('savefig_example.png')
plt.show()
