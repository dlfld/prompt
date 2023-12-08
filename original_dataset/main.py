def merge_ctb_data():
    datasets = ["ctb/dev.txt", "ctb/test.txt", "ctb/train.txt"]
    all_data = []
    for item in datasets:
        all_data.extend(read_file(item))
    print(len(all_data))
    with open("../code/data/ctb/totaldata.txt", "w") as f:
        f.writelines(all_data)


def read_file(file):
    with open(file, "r") as f:
        res = f.readlines()
        return res


if __name__ == '__main__':
    merge_ctb_data()
