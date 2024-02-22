import matplotlib.pyplot as plt
plt.style.use('ggplot')
# 处理中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置字体为宋体
plt.rcParams['font.size'] = 12  # 设置字号为12
# 数据
models = ['BP+CH', 'BP+C', 'BP', 'MBP+CH', 'MBP+C', 'MBP', 'BAP+CH', 'BAP+C', 'BAP']
shot_sizes = [5, 10, 15, 20, 25]
f1_values = {
    'BP+CH': [61.36, 73.60, 78.63, 81.13, 82.57],
    'BP+C': [59.91, 72.02, 78.36, 80.93, 81.98],
    'BP': [59.62, 70.58, 76.46, 80.15, 81.04],
    'MBP+CH': [64.07, 75.26, 81.11, 84.16, 84.65],
    'MBP+C': [58.30, 74.66, 79.47, 81.46, 82.62],
    'MBP': [62.00, 73.60, 80.19, 81.93, 82.69],
    'BAP+CH': [64.51, 79.76, 82.52, 83.68, 85.52],
    'BAP+C': [61.25, 76.52, 80.07, 82.03, 84.31],
    'BAP': [60.96, 75.72, 80.27, 82.46, 83.57]
}

# 定义颜色
bp_colors = ['red', 'green', 'blue']

# 设置子图
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# 第一个子图：BP系列
bp_models = [model for model in models if model.startswith('BP')]
for model, color in zip(bp_models, bp_colors):
    axs[0].plot(shot_sizes, f1_values[model], marker='o', linestyle='-', label=model, color=color)
axs[0].set_title('BP系列模型的F1值变化')
axs[0].set_xlabel('Shot大小')
axs[0].set_ylabel('F1平均值')
axs[0].legend()
axs[0].grid(True)

# 第二个子图：MBP系列
mbp_models = [model for model in models if model.startswith('MBP')]
for model, color in zip(mbp_models, bp_colors):
    axs[1].plot(shot_sizes, f1_values[model], marker='o', linestyle='-', label=model, color=color)
axs[1].set_title('MBP系列模型的F1值变化')
axs[1].set_xlabel('Shot大小')
axs[1].set_ylabel('F1平均值')
axs[1].legend()
axs[1].grid(True)

# 第三个子图：BAP系列
bap_models = [model for model in models if model.startswith('BAP')]
for model, color in zip(bap_models, bp_colors):
    axs[2].plot(shot_sizes, f1_values[model], marker='o', linestyle='-', label=model, color=color)
axs[2].set_title('BAP系列模型的F1值变化')
axs[2].set_xlabel('Shot大小')
axs[2].set_ylabel('F1平均值')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
