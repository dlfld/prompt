import os


def read_file(filename):
    lines = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "").strip()
            if len(line) > 0:
                lines.append(float(line))
    return lines


import matplotlib.pyplot as plt


def craw_2(datas, labels, train_size, mode):
    # plt.figure(figsize=(20, 20), dpi=300)
    for j in range(len(datas)):
        for i in range(len(datas[j])):
            plt.plot([x for x in range(len(datas[j][i]))], datas[j][i], label=labels[j][i])

    plt.title(f'train_size={train_size},mode={mode} loss figure')
    plt.legend(loc='lower right')

    plt.legend()
    plt.show()
    # plt.savefig(f"{train_size}_{mode}.png")


def get_fig_data(train_size, mode):
    dir_names = ["p-tuning/"]
    datas = []
    labels = []
    for dir_name in dir_names:
        dir_list = os.listdir(dir_name)
        pre_name = dir_name.replace("ud_ch_", "").replace("/", "")
        for item in dir_list:
            file_dir = dir_name + item
            if item.endswith(f"_{train_size}_1_{mode}.csv"):
                train_loss = read_file(file_dir)
                datas.append(train_loss)
                label_name = pre_name + "_" + item
                labels.append(label_name)

    return datas, labels


if __name__ == '__main__':
    # train_sizes = [5, 10, 15, 20, 25, 50, 75, 100, 200, 500]
    train_sizes = [5]
    for train_size in train_sizes:
        modes = ["train", "test"]
        datas, labels = [], []
        for mode in modes:
            data, label = get_fig_data(train_size, mode)
            datas.append(data)
            labels.append(label)
        craw_2(datas, labels, train_size, "BERT")
