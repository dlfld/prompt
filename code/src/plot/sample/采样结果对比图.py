import joblib

from sample import NER_Adaptive_Resampling


def get_label_freq(data):
    res = dict({})
    for item in data:
        labels = item[1].split('/')
        for label in labels:
            if label not in res.keys():
                res[label] = 1
            else:
                res[label] += 1
    return res
def print_res(label_map):
    res = []
    for item in label_map.keys():
        item = [item,label_map[item]]
        res.append(item)
    print(res)

if __name__ == '__main__':
    path = "/Users/dailinfeng/Desktop/prompt/code/data/split_data/fold/{item}.data"
    shots = [5, 10, 15, 20, 25]
    datas = joblib.load(path.format(item=25))

    for data in datas:
        before_freq = get_label_freq(data)
        sample_res = NER_Adaptive_Resampling(data).resamp("nsCRD")
        print(len(sample_res))
        after_freq = get_label_freq(sample_res)
        print(len(data))

        print_res(before_freq)
        print_res(after_freq)

