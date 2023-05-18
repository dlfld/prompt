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


def generate_prompt(sentence :str,word:str,pre_part_of_speech:str,pre_word:str,part_of_speech:str)->str:
    """
    :param sentence: 句子
    :param word: 当前的词语
    :param pre_part_of_speech: 前一个词语的词性
    :param pre_word: 前一个词语
    :param part_of_speech: 当前词语的词性
    """
    template = "在句子“{sentence}”中，词语“{word}”的前文如果是由“{pre_part_of_speech}”词性的词语“{pre_word}”来修饰，那么词语“{word}”的词性是“[MASK]”→ {part_of_speech}"
    return template.format(sentence=sentence,word=word,pre_part_of_speech=pre_part_of_speech,pre_word=pre_word,part_of_speech=part_of_speech)

def generate_data_seq(datas: List[List[str]]):
    """
        依次生成prompt并进行预测
    :param datas:输入的数据
    [
        ['脉/数', 'NR/VA'],
        ...
    ]
    """
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
            # 生成输入模型的pair
            cur_data = generate_prompt(sentence=cur_sentence,word=cur_word,pre_part_of_speech=pre_part_of_speech,pre_word=pre_word,part_of_speech=cur_part_of_speech)




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
            
            # 添加当前单词
            word_set.add(cur_word+"\n")
            # 添加当前label
            # word_set.add(cur_part_of_speech)

            label_set.add(cur_part_of_speech)
            prompt = template.format(cur_sentence, cur_word, pre_part_of_speech, pre_word, cur_word, cur_part_of_speech)
            data["prompts"].append(prompt)

        dataset.append(data)
    return dataset,label_set,word_set

def get_cur_vocab(datas: List[List[str]]) -> List[Dict[str, Any]]:
    """
        获取当前数据集的词表
    :param datas:
    [
        ['脉/数', 'NR/VA'],
        ...
    ]
    :return:
    """
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
            # 当前词性
            cur_part_of_speech = label[index]
            
            # 添加当前单词
            # word_set.add(cur_word)
            # 添加当前label
            word_set.add(cur_part_of_speech)
    return word_set


def add_cur_token_into_vocab():
    """
        读取当前数据，获取数据的所有唯一token，将token加入到词表的后面
    """
    # 原数据路径
    origin_data_path = "/home/dlf/prompt/code/data/jw/pos_seg.txt"  
    # 当前模型vocab.txt 词表路径
    model_vocab_path = "/home/dlf/prompt/code/model/bert_large_chinese/vocab.txt"
    # 新词表路径
    vocab_appended_path = "/home/dlf/prompt/vocab.txt"

    # 读取初始数据
    datas = data_reader(origin_data_path)
    # 转换为标准数据
    standard_data = format_data_type(datas)
    # 获取当前词表数据 
    token_set = get_cur_vocab(standard_data)
    return token_set
    # for item in list(token_set):
    #     print(item.replace('\n', ""))
    # print(len(token_set))
    # 读取当前模型vocab.txt 词表
    # cur_vocab = data_reader(model_vocab_path)
   
    # 在不改变词表顺序的情况下向词表中添加原本不存在的元素
    # for token in token_set:
    #     if token not in cur_vocab:
    #         cur_vocab.append(token)

    # 写入新词表
    # with open(vocab_appended_path,"w") as f:
    #     f.writelines(cur_vocab)
    
    

if __name__ == '__main__':
    # 读取数据的token，将它添加到vocab的后面
    # res  =  add_cur_token_into_vocab()
    # res = list(res)
    # print(res)
    # 在句子“脉细弱”中，词语“脉”的前文如果是由“[CLS]”词性的词语“[CLS]”来修饰，那么词语“脉”的词性是“[MASK]”, NR
    res = generate_prompt(
        sentence="脉细弱",
        word="脉",
        pre_part_of_speech="[CLS]",
        pre_word="[CLS]",
        part_of_speech="NR"
    )
    print(res)
    # 生成csv的数据集
    # with open("dataset.csv","w") as f:
    #     import csv
    #     csv_writer = csv.writer(f)
    #     # f.writelines(list(dataset))
    #     csv_writer.writerow(["text","label"])
    #     csv_writer.writerows(list(dataset))
    
