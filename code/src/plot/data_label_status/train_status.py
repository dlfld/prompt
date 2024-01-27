import copy

import joblib


def draw_bar(label_freq):
    import matplotlib.pyplot as plt
    # 设置绘图风格
    plt.style.use('ggplot')
    # 处理中文乱码
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 假设label_freq是一个包含标签及其频次的字典
    # label_freq = {'Label1': 20, 'Label2': 30, 'Label3': 15, 'Label4': 25}

    # 提取标签和频次
    labels = list(label_freq.keys())
    frequencies = list(label_freq.values())

    # 绘制柱状图
    plt.bar(labels, frequencies, color='blue')

    # 添加标签和标题
    # plt.xlabel('标签')
    # plt.ylabel('频次')
    # plt.title('每个标签的频次')

    # 显示图表
    plt.show()


if __name__ == '__main__':
    all_labels = ["NR", "NN", "AD", "PN", "OD", "CC", "DEG",
                  "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
                  "VA", "VE"]
    all_labels_map = {key:0 for key in all_labels}
    data_all = joblib.load("/Users/dailinfeng/Desktop/prompt/code/data/split_data/fold/25.data")
    num_labels = []

    for fold in data_all:
        print(fold)
        labels_map = copy.deepcopy(all_labels_map)
        for data in fold:
            labels = data[1].split("/")
            for label in labels:
                if label in labels_map.keys():
                    labels_map[label] += 1
                else:
                    labels_map[label] = 1
        # print(len(labels_map.keys()))
        # print(labels_map)
        labels_map = dict(sorted(labels_map.items(), key=lambda item: item[1], reverse=True))
        draw_bar(labels_map)
        num_label = 0
        for k in labels_map.keys():
            if labels_map[k] > 0:
                num_label+=1
        num_labels.append(num_label)

    print(num_labels)
