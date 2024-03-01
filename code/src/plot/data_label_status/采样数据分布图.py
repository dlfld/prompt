import matplotlib.pyplot as plt

# 设置绘图风格
plt.style.use('ggplot')
# 处理中文乱码
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置字体为宋体
plt.rcParams['font.size'] = 12  # 设置字号为12
# 采样后的标签分布数据
post_sampling_data = {
    'NN': 8667, 'VV': 5334, 'PU': 5010, 'AD': 3126, 'NR': 1632, 'DEG': 978,
    'CD': 957, 'M': 857, 'PN': 817, 'JJ': 779, 'VA': 573, 'VC': 513,
    'LC': 496, 'CC': 464, 'VE': 305, 'SP': 185, 'OD': 81
}

# 采样前的标签分布数据
pre_sampling_data = {
    'NN': 5347, 'PU': 3386, 'VV': 3282, 'AD': 2182, 'NR': 1176, 'DEG': 661,
    'PN': 610, 'CD': 600, 'M': 525, 'JJ': 516, 'VA': 450, 'VC': 396,
    'LC': 307, 'CC': 282, 'SP': 215, 'VE': 199, 'OD': 55
}

# 设置柱子的宽度
bar_width = 0.35

# 设置标签和数据
labels = list(post_sampling_data.keys())
post_sampling_values = list(post_sampling_data.values())
pre_sampling_values = list(pre_sampling_data.values())

# 计算每组柱子的位置
index = range(len(labels))
pre_index = [i - bar_width / 2 for i in index]
post_index = [i + bar_width / 2 for i in index]

# 创建图表
plt.figure(figsize=(14, 8))

# 绘制柱状图
plt.bar(pre_index, pre_sampling_values, bar_width, label='采样前', color='darkorange')
plt.bar(post_index, post_sampling_values, bar_width, label='采样后', color='darkblue')

# 设置标题和标签
plt.title('采样前后的标签分布情况对比')
plt.xlabel('标签')
plt.ylabel('频次')
plt.xticks(index, labels, rotation=45)  # 设置x轴标签并旋转以防重叠
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
