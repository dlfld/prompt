from typing import List
import copy
import sys

sys.path.append("..")
import logddd
import torch
from datasets import DatasetDict
from data_process.pos_seg_2_standard import format_data_type_pos_seg
from data_process.news_data_2_standard import format_data_type_people_daily
from data_process.utils import data_reader
from data_process.ctb_data_2_standard import get_all_ctb_data, format_data_type_ctb


def load_data(data_files: str) -> List[List[str]]:
    """
        根据数据集的位置，加载数据集，并将数据集处理成标准格式
        :param data_files:数据集位置
        :return : 标准格式
    """
    # 读取初始数据
    datas = data_reader(data_files)
    # 去除掉第一行的 列名
    # datas = datas[1:]
    # 转换为标准数据
    standard_data = format_data_type_pos_seg(datas)
    # standard_data = format_data_type_people_daily(datas)
    return standard_data


def load_ctb_data() -> List[List[str]]:
    all_data = get_all_ctb_data()
    standard_data = format_data_type_ctb(all_data)
    return standard_data


def load_instance_data(standard_data: List[List[str]], tokenizer, Config, is_train_data: bool):
    """
      加载训练用的数据集
      :param standard_data: 标准格式的数据
      :param tokenizer:tokenizer对象
      :param Config: 模型配置类
      :param is_train_data: 是否是加载train数据，train 和test的数据加载有一定的偏差
    """
    # 每一条数据转换成的prompt列表 [[prompts],[prompts],...]
    instance_data = []
    from tqdm.auto import tqdm

    for data in tqdm(standard_data, desc="load_instance_data:"):
        # 将一条数据转换成一系列的prompts
        prompts = build_a_list_of_prompts_not_split([data], is_train_data)
        # 遍历每一个prompt，将其转换为可以直接输入模型的数据
        prompt_texts = []
        prompt_labels = []
        for prompt in prompts:
            prompt_texts.append(prompt[0])
            prompt_labels.append(prompt[1])

        result = tokenizer(prompt_texts, return_tensors="pt", padding="max_length", max_length=Config.sentence_max_len)
        result["labels"] = [tokenizer.convert_tokens_to_ids(str(label).strip().replace("\n", "")) for label in
                            prompt_labels]
        # 保存当前列的label
        label = copy.deepcopy(result["labels"])
        # 复制当前label过去
        labels = copy.deepcopy(result["input_ids"])
        for index, sentence_words in enumerate(result["input_ids"]):
            # 遍历当前句子的每一个word
            for word_index, word in enumerate(sentence_words):
                # 当前word 的id如果是等于mask的id
                if word == tokenizer.mask_token_id:
                    # 第index个句子，第word_index个词的id为 = 第index个label
                    # result["labels"][index][word_index] = label[index]
                    labels[index][word_index] = label[index]
                else:
                    labels[index][word_index] = -100

        result["labels"] = labels
        # 删除不需要的key token_type_ids
        del result["token_type_ids"]

        instance_data.append(result)

    return instance_data


def generate_prompt(sentence: str, word: str, pre_part_of_speech: str, pre_word: str, part_of_speech: str) -> str:
    """
        生成一个prompt句子
    :param sentence: 句子
    :param word: 当前的词语
    :param pre_part_of_speech: 前一个词语的词性
    :param pre_word: 前一个词语
    :param part_of_speech: 当前词语的词性
    """
    template = "在句子“{sentence}”中，词语“{word}”的前文如果是由“{pre_part_of_speech}”词性的词语“{pre_word}”来修饰，那么词语“{word}”的词性是“[MASK]”→ {part_of_speech}"
    return template.format(sentence=sentence, word=word, pre_part_of_speech=pre_part_of_speech, pre_word=pre_word,
                           part_of_speech=part_of_speech)


def build_a_list_of_prompts_not_split(datas: List[List[str]], is_train_data: bool) -> List[List[str]]:
    # 数据集
    """
        生成不按照具体划分的数据集
        :param datas: 输入是标准的数据集
        :param is_train_data: 是否是加载train数据，train 和test的数据加载有一定的偏差
        :return  [
                    [data,label]
                ] 输出
    """
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
            pre_part_of_speech = "[CLS]" if index == 0 else label[index - 1] if is_train_data else "[PLB]"
            # 前文词语
            pre_word = "[CLS]" if index == 0 else sentence[index - 1]
            # 当前词语
            cur_word = word[0]
            # 当前词性
            cur_part_of_speech = label[index]
            # 生成输入模型的pair
            prompt = generate_prompt(sentence=cur_sentence, word=cur_word, pre_part_of_speech=pre_part_of_speech,
                                     pre_word=pre_word, part_of_speech=cur_part_of_speech)
            # logddd.log(prompt)
            dataset.append([prompt.split("→")[0], prompt.split("→")[1].strip()])

    return dataset


def get_cur_vocab(datas: List[List[str]]) -> List[str]:
    """
        获取当前数据集的词表 。填充词表的时候用
    :param datas:
    [
        ['脉/数', 'NR/VA'],
        ...
    ]
    :return: 当前词表的词或者label
    """
    # 词表
    word_set = set()

    # 遍历整个数据集
    for item in datas:
        # 进行条数据生成
        sentence = item[0].split("/")
        label = item[1].split("/")
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
    standard_data = format_data_type_pos_seg(datas)
    # 获取当前词表数据 
    token_set = get_cur_vocab(standard_data)
    return token_set


def generate_dataset_csv():
    # 原数据路径
    origin_data_path = "/home/dlf/prompt/code/data/jw/pos_seg.txt"
    # 读取初始数据
    datas = data_reader(origin_data_path)
    # 转换为标准数据
    standard_data = format_data_type_pos_seg(datas)
    # 将标准数据转换为输入prompt的训练数据
    dataset = build_a_list_of_prompts_not_split(standard_data)
    # 保存数据
    with open("dataset.csv", "w") as f:
        import csv
        csv_writer = csv.writer(f)
        # f.writelines(list(dataset))
        csv_writer.writerow(["text", "label"])
        csv_writer.writerows(list(dataset))
