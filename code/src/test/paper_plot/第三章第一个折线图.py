import matplotlib.pyplot as plt

# 数据
models = ['BP+CH', 'MBP+CH', 'BAP+CH']
shot_sizes = [5, 10, 15, 20, 25]
f1_values = {
    'BP+CH': [61.36, 73.60, 78.63, 81.13, 82.57],
    'MBP+CH': [64.07, 75.26, 81.11, 84.16, 84.65],
    'BAP+CH': [64.51, 79.76, 82.52, 83.68, 85.52]
}

# 定义颜色
colors = ['darkorange', 'green', 'navy']

# 创建图表
plt.figure(figsize=(10, 6))
# 设置绘图风格
plt.style.use('ggplot')
# 处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置字体为宋体
plt.rcParams['font.size'] = 12  # 设置字号为12
# 绘制每个模型的F1值变化曲线，并调整标注位置
for idx, model in enumerate(models):
    plt.plot(shot_sizes, f1_values[model], marker='o', linestyle='-', label=model, color=colors[idx])
    for i, (x, y) in enumerate(zip(shot_sizes, f1_values[model])):
        if model == 'MBP+CH':
            if i == 3:  # 第四个点的数据（索引为3）
                plt.text(x, y - 1.5, f'{y:.2f}', ha='left', va='center', color=colors[idx], fontsize=9, fontweight='bold')
            else:
                plt.text(x + 0.5, y, f'{y:.2f}', ha='left', va='center', color=colors[idx], fontsize=9, fontweight='bold')
        elif model == 'BAP+CH':
            plt.text(x, y + 1, f'{y:.2f}', ha='center', va='bottom', color=colors[idx], fontsize=9, fontweight='bold')
        else:  # BP+CH
            if i == 0:  # 第一个点的数据（索引为0）
                plt.text(x, y + 1, f'{y:.2f}', ha='center', va='bottom', color=colors[idx], fontsize=9, fontweight='bold')
            else:
                plt.text(x, y - 1, f'{y:.2f}', ha='center', va='top', color=colors[idx], fontsize=9, fontweight='bold')

plt.title('BP+CH, MBP+CH, BAP+CH模型的F1值变化曲线')
plt.xlabel('Shot大小')
plt.ylabel('F1值')
plt.xticks(shot_sizes)
plt.legend()

plt.grid(True)
plt.show()
