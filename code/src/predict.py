from typing import List

import math


from data_process.data_processing import generate_prompt
from data_process.utils import data_reader

from data_process.pos_seg_2_standard import format_data_type_pos_seg

import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import logddd

def link_predict(model,tokenizer,epoch,Config):
    """
        使用模型进行链式的预测
        :param model:   模型
        :param tokenizer: tokenizer对象
        :param epoch: epoch
        
    """

    res_file_path = Config.predict_res_file.format(epoch)

    # 读取初始数据
    datas = data_reader(Config.test_dataset_path)
    # 转换为标准数据
    standard_data = format_data_type_pos_seg(datas)
    # 所有数据进行链式预测的结果文件
    all_data_predict_res = []
    for data in standard_data:
        # 对一条数据进行预测，一条数据指的是一个句子，这个句子可以生成len(sentence)个prompt
        sentence_predict_res = generate_data_seq(data,model,tokenizer)
        all_data_predict_res.extend(sentence_predict_res)
        # 添加空行 隔开
        all_data_predict_res.append("\n")

    save_predict_file(res_file_path,all_data_predict_res)


def save_predict_file(file_path:str,content:List[str]):
   """
    存储预测结果文件
    :param file_path: 结果文件
    :param content: 内容
   """
   with open(file_path,"w",encoding="utf-8") as f:
       f.writelines(content)

def generate_data_seq(item: List[str],model,tokenizer)->List[str]:
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
        cur_data = generate_prompt(sentence=cur_sentence,word=cur_word,pre_part_of_speech=pre_part_of_speech,pre_word=pre_word,part_of_speech=cur_part_of_speech)

        # ================对sequence进行预测====================
        # data切分
        data = cur_data.split("→")[0].strip()
        # label切分
        cur_seq_label = cur_data.split("→")[1].strip().replace("\n","")
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
        

def test_model(model,dataloader,tokenizer,epoch,Config):
    """
        :param model: 模型
        :param dataloader: 数据集 dataloader
        :param tokenizer: tokenizer
        :param epoch: 当前的轮数
    """
    model.eval()

    losses = []
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss)
    t_losses = torch.tensor(losses)

    # logddd.log(len(t_losses))
    # print(Config.train_size)
    t_losses = t_losses[: Config.train_size]
    try:
        perplexity = math.exp(torch.mean(t_losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    with torch.no_grad():
        # 链式调用预测
        link_predict(model,tokenizer,epoch,Config)

        
if __name__ == '__main__':
    model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    link_predict(model,tokenizer)

           
            