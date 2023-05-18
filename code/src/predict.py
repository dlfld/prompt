from typing import List
from data_processing import generate_prompt, data_reader, format_data_type
import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

def link_predict(model,tokenizer):
    """
        使用模型进行链式的预测
        :param model:   模型
        :param tokenizer: tokenizer对象
        
    """
    # 原数据路径
    # origin_data_path = "/home/dlf/prompt/code/data/jw/pos_seg.txt"  
    origin_data_path = "/home/dlf/prompt/code/data/jw/short_data.txt"

    # 读取初始数据
    datas = data_reader(origin_data_path)
    # 转换为标准数据
    standard_data = format_data_type(datas)
    for data in standard_data:
        # 对一条数据进行预测
        generate_data_seq(data,model,tokenizer)

        
        
    
    
def generate_data_seq(item: List[str],model,tokenizer):
    """
        依次生成prompt并进行预测
        :param item:输入的数据
        :param model:   模型
        :param tokenizer: tokenizer对象
        ['脉/数', 'NR/VA']
    """
    # 进行条数据生成
    sentence = item[0].split("/")
    label = item[1].split("/")
    # 前文词性
    pre_part_of_speech = "[CLS]" 
    for index, word in enumerate(zip(sentence, label)):
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
        # 对sequence进行预测
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
        # print(predict_label)
        # 保存当前的预测结果
        pre_part_of_speech = predict_label
        print(f"'{data.replace(tokenizer.mask_token, predict_label)}' -> 正确值为： {cur_seq_label}" )
        
        
        
if __name__ == '__main__':
    model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    link_predict(model,tokenizer)

           
            