from typing import List

import math
from sklearn.metrics import classification_report
from utils import get_prf
from model_params import Config
from data_process.data_processing import generate_prompt
from data_process.utils import data_reader

from data_process.pos_seg_2_standard import format_data_type_pos_seg

import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import logddd


def link_predict(model, tokenizer, epoch):
    """
        使用模型进行链式的预测
        :param model:   模型
        :param tokenizer: tokenizer对象
        :param epoch: epoch
        
    """
    # 结果文件保存路径
    res_file_path = Config.predict_res_file.format(epoch)
    # # 读取初始数据
    # datas = data_reader(Config.test_dataset_path)
    # 转换为标准数据
    standard_data = load_data(res_file_path)
    labels = []
    y_preds = []
    for data in standard_data:
        # 对一条数据进行预测，一条数据指的是一个句子，这个句子可以生成len(sentence)个prompt
        label,path = generate_data_seq_viterbi(data, model, tokenizer)
        labels.extend(label)
        y_preds.extend(path)


    report = classification_report(labels, y_preds)
    print(report)
    print()

    # 保存预测结果
    # save_predict_file(res_file_path, all_data_predict_res)


def save_predict_file(file_path: str, content: List[str]):
    """
    存储预测结果文件
    :param file_path: 结果文件
    :param content: 内容
   """
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(content)


def generate_data_seq(item: List[str], model, tokenizer) -> List[str]:
    """
        对一个句子依次生成prompt，并按链式方式进行调用
        :param item:输入的数据
        :param model:   模型
        :param tokenizer: tokenizer对象
        ['脉/数', 'NR/VA']
        :return 生成prompt之后并记录下预测的结果，返回预测结果列表
    """
    # 进行条数据生成
    sentence = item[0].split("/")
    label = item[1].split("/")
    # 预测结果列表
    predict_res_list = []
    # 前文词性
    pre_part_of_speech = "[CLS]"
    for index, word in enumerate(zip(sentence, label)):
        # ================对template中的空进行处理====================
        # 当前句子 '脉/弦/大' -> 脉弦大
        cur_sentence = item[0].replace("/", "")
        # 前文词语
        pre_word = "[CLS]" if index == 0 else sentence[index - 1]
        # 当前词语
        cur_word = word[0]
        # 当前词性
        cur_part_of_speech = label[index]
        # 生成输入模型的pair，生成一个sequence
        cur_data = generate_prompt(sentence=cur_sentence, word=cur_word, pre_part_of_speech=pre_part_of_speech,
                                   pre_word=pre_word, part_of_speech=cur_part_of_speech)

        # ================对sequence进行预测====================
        # data切分
        data = cur_data.split("→")[0].strip()
        # label切分
        cur_seq_label = cur_data.split("→")[1].strip().replace("\n", "")
        inputs = tokenizer(data, return_tensors="pt")
        token_logits = model(**inputs).logits
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
        # 当前预测出来的token
        token = top_5_tokens[0]
        # 预测出来的label
        predict_label = tokenizer.decode([token])
        # 保存当前的预测结果 下一条会用到当前的label
        pre_part_of_speech = predict_label
        predict_res = f"'{data.replace(tokenizer.mask_token, predict_label)}' -> 正确值为： {cur_seq_label}\n"
        print(predict_res)
        predict_res_list.append(predict_res)

    return predict_res_list

def build_a_list_of_prompts_not_split(datas: List[List[str]]) -> List[List[str]]:
    # 数据集
    """
        生成不按照具体划分的数据集
        :param datas: 输入是标准的数据集
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
            pre_part_of_speech = "[CLS]" if index == 0 else "[PLB]"
            # 前文词语
            pre_word = "[CLS]" if index == 0 else sentence[index - 1]
            # 当前词语
            cur_word = word[0]
            # 当前词性
            cur_part_of_speech = label[index]
            # 生成输入模型的pair
            prompt = generate_prompt(sentence=cur_sentence, word=cur_word, pre_part_of_speech=pre_part_of_speech,pre_word=pre_word, part_of_speech=cur_part_of_speech)
            # logddd.log(prompt)
            dataset.append([prompt.split("→")[0], prompt.split("→")[1].strip()])

    return dataset

def generate_data_seq_viterbi(item: List[str], model, tokenizer) -> (List[int],List[int]):
    """
        对一个句子依次生成prompt，并按链式方式进行调用
        :param item:输入的数据         ['脉/数', 'NR/VA']
        :param model:   模型
        :param tokenizer: tokenizer对象

        :return 生成prompt之后并记录下预测的结果，返回预测结果列表
    """
    # 将一条数据转换成一系列的prompts
    prompts = build_a_list_of_prompts_not_split([item])
    # 遍历每一个prompt，将其转换为可以直接输入模型的数据
    prompt_texts = []
    # 记录当前的真实lanel
    labels = []
    prompt_labels = []
    for prompt in prompts:
        prompt_texts.append(prompt[0])
        labels.append(tokenizer.convert_tokens_to_ids(prompt[1]))
        prompt_labels.append(prompt[1])

        result = tokenizer(prompt_texts, return_tensors="pt", padding="max_length", max_length=Config.sentence_max_len)
        result["labels"] = [tokenizer.convert_tokens_to_ids(str(label).strip().replace("\n", "")) for label in prompt_labels]
        # Create a new labels column
        # 保存当前列的label
        label = copy.deepcopy(result["labels"])
        # 复制当前label过去
        labels = copy.deepcopy(result["input_ids"])
        for index, sentence_words in enumerate(result["input_ids"]):
            # 遍历当前句子的每一个word
            for word_index, word in enumerate(sentence_words):
                # 当前word 的id如果是等于mask的id
                if word == tokenizer.mask_token_id:
                    # print("进来了")
                    # 第index个句子，第word_index个词的id为 = 第index个label
                    # result["labels"][index][word_index] = label[index]
                    labels[index][word_index] = label[index]
                else:
                    labels[index][word_index] = -100

        result["labels"] = labels
    # 删除不需要的key token_type_ids
    del result["token_type_ids"]

    result = {
        k:v.tolist()
        for k,v in result.items()
    }
    total_path, _,_= model(result)
    total_path = [x + 1 for x in total_path]
    # print(item[0])
    # print("预测序列", tokenizer.convert_ids_to_tokens(total_path))
    # print("实际序列", item[1].split("/"))
    #
    # print()

    return labels,total_path


def generate_data_seq_viterbi_(item: List[str], model, tokenizer) -> List[str]:
    """
        对一个句子依次生成prompt，并按链式方式进行调用
        :param item:输入的数据         ['脉/数', 'NR/VA']
        :param model:   模型
        :param tokenizer: tokenizer对象

        :return 生成prompt之后并记录下预测的结果，返回预测结果列表
    """
    # 进行条数据生成
    sentence = item[0].split("/")
    label = item[1].split("/")
    # 预测结果列表
    predict_res_list = []
    # 前文词性
    pre_part_of_speech = "[CLS]"
    for index, word in enumerate(zip(sentence, label)):
        # ================对template中的空进行处理====================
        # 当前句子 '脉/弦/大' -> 脉弦大
        cur_sentence = item[0].replace("/", "")
        # 前文词语
        pre_word = "[CLS]" if index == 0 else sentence[index - 1]
        # 当前词语
        cur_word = word[0]
        # 当前词性
        cur_part_of_speech = label[index]
        # 生成输入模型的pair，生成一个sequence
        cur_data = generate_prompt(sentence=cur_sentence, word=cur_word, pre_part_of_speech=pre_part_of_speech,
                                   pre_word=pre_word, part_of_speech=cur_part_of_speech)

        # ================对sequence进行预测====================
        # data切分
        data = cur_data.split("→")[0].strip()
        # label切分
        cur_seq_label = cur_data.split("→")[1].strip().replace("\n", "")
        inputs = tokenizer(data, return_tensors="pt", padding="max_length", max_length=Config.sentence_max_len)

        # 输入数据所有的路径和分数
        total_path, _ = model([[inputs]])
        total_path = int(total_path[0][0])
        pre_part_of_speech = tokenizer.convert_ids_to_tokens(total_path)
        predict_res = f"'{data.replace(tokenizer.mask_token, pre_part_of_speech)}' -> 正确值为： {cur_seq_label}"
        logddd.log(predict_res)

    return predict_res_list


def test_model(model, dataloader, tokenizer, epoch):
    """
        加载验证集 链式预测
        :param model: 模型
        :param dataloader: 数据集 dataloader
        :param tokenizer: tokenizer
        :param epoch: 当前的轮数
    """
    model.eval()

    with torch.no_grad():
        # 链式调用预测
        link_predict(model, tokenizer, epoch)


if __name__ == '__main__':
    model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    link_predict(model, tokenizer,1)
