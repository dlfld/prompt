from typing import List, Any, Dict


def data_reader(filename: str) -> List[str]:
    """
        读取文件，以行为单位返回文件内容
    :param filename: 文件路径
    :return: 以行为单位的内容list
    """
    with open(filename, "r") as f:
        return f.readlines()


def format_data_type(datas: List[str]) -> List[List[str]]:
    """
        对数据格式进行更改，脉 数    NR VA    10592 -> ['脉/数', 'NR/VA']
    :param datas: eg: 脉 沉 细    NR VA VA    10138
    :return: 更改好的数据
    [
        ['脉/数', 'NR/VA'],
        ...
    ]
    """
    res = []
    for data in datas:
        # 脉 数    NR VA    10592 -> 脉 数 && NR VA && 10592
        data = data.replace("    ", "&&")
        # 脉 数&&NR VA&&10592 -> 脉/数&&NR/VA&&10592
        data = data.replace(" ", "/")
        # 脉/数&&NR/VA&&10592 -> ['脉/数', 'NR/VA']
        data = data.split("&&")[:-1]
        res.append(data)
    return res
def build_a_list_of_prompts_not_split(datas: List[List[str]]) -> List[Dict[str, Any]]:
    template = '在句子“{}”中，词语“{}”的前文如果是由“{}”词性的词语“{}”来修饰，那么词语“{}”的词性是“[MASK]”→ {}'
    # 数据集
    dataset = []

    # 遍历整个数据集
    for item in datas:
        # 进行条数据生成
        sentence = item[0].split("/")
        label = item[1].split("/")

        for index, word in enumerate(zip(sentence, label)):
            # 当前句子 '脉/弦/大' -> 脉弦大
            cur_sentence = item[0].replace("/", "")
            # 前文词性
            pre_part_of_speech = "[CLS]" if index == 0 else label[index - 1]
            # 前文词语
            pre_word = "[CLS]" if index == 0 else sentence[index - 1]
            # 当前词语
            cur_word = word[0]
            # 当前词性
            cur_part_of_speech = label[index]
            prompt = template.format(cur_sentence, cur_word, pre_part_of_speech, pre_word, cur_word, cur_part_of_speech)
            dataset.append([prompt.split("→")[0],prompt.split("→")[1]])

    return dataset

def build_a_list_of_prompts(datas: List[List[str]]) -> List[Dict[str, Any]]:
    """
        构建为prompt数据
    :param datas:
    [
        ['脉/数', 'NR/VA'],
        ...
    ]
    :return:
    """
    template = "在句子“{}”中，词语“{}”的前文如果是由“{}”词性的词语“{}”来修饰，那么词语“{}”的词性是“[MASK]”→ {}"
    # 数据集
    dataset = []
    label_set = set()
    # 词表
    word_set = set()

    # 遍历整个数据集
    for item in datas:
        # 进行条数据生成
        sentence = item[0].split("/")
        label = item[1].split("/")
        data = {
            "origin_sentence": f"{item[0]},{item[1]}",
            "prompts": []
        }
        for index, word in enumerate(zip(sentence, label)):
            # 当前句子 '脉/弦/大' -> 脉弦大
            cur_sentence = item[0].replace("/", "")
            # 前文词性
            pre_part_of_speech = "[CLS]" if index == 0 else label[index - 1]
            # 前文词语
            pre_word = "[CLS]" if index == 0 else sentence[index - 1]
            # 当前词语
            cur_word = word[0]
            # 当前词性
            cur_part_of_speech = label[index]

            word_set.add(cur_word+"\n")
            label_set.add(cur_part_of_speech)
            prompt = template.format(cur_sentence, cur_word, pre_part_of_speech, pre_word, cur_word, cur_part_of_speech)
            data["prompts"].append(prompt)

        dataset.append(data)
    return dataset,label_set,word_set


from sklearn.model_selection import train_test_split
def split_data(data):
    """
        没有用dataloader的原因是：现在模型是一条一条数据输入的，数据长度没有对齐
        划分数据，换分训练集测试集和验证集
        @return 训练集、验证集、测试集
    """
    X_train, X_validate_test, _, _ = train_test_split(
        data, data, test_size=0.2, random_state=42)
    X_validate, X_test, _, _ = train_test_split(
        X_validate_test, X_validate_test, test_size=0.5, random_state=42)
    return X_train, X_validate, X_test


def get_all_data():
    """
        获取所有数据
    """
    # 读取初始数据
    datas = data_reader("/home/dlf/crf/code/data/jw/pos_seg.txt")
    # 转换为标准数据
    standard_data = format_data_type(datas)
    # dataset,label_set = build_a_list_of_prompts(standard_data)
    dataset = build_a_list_of_prompts_not_split(standard_data)
    return split_data(dataset)

if __name__ == '__main__':
    # 读取初始数据
    datas = data_reader("/home/dlf/crf/code/data/jw/pos_seg.txt")
    # 转换为标准数据
    standard_data = format_data_type(datas)
    # dataset,label_set,word_set = build_a_list_of_prompts(standard_data)
    dataset = build_a_list_of_prompts_not_split(standard_data)
    # print(len(word_set))
    with open("dataset.csv","w") as f:
        import csv
        csv_writer = csv.writer(f)
        # f.writelines(list(dataset))
        csv_writer.writerow(["text","label"])
        csv_writer.writerows(list(dataset))
    
    # return dataset
    # for item in dataset:
    #     print(item)
    # for item in label_set:
    #     print(item)