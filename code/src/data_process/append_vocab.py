
"""
    在当前现有的词表中添加当前数据集的token，然后重构词汇矩阵。

"""
from data_processing import add_cur_token_into_vocab

if __name__ == '__main__':
    # 在现有的词表中添加进去当前数据集的token，并重构词汇矩阵
    new_tokens = add_cur_token_into_vocab()
    new_tokens = list(new_tokens)
    from transformers import BertForMaskedLM, BertTokenizer

    model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = BertForMaskedLM.from_pretrained(model_checkpoint)

    num_added_toks = tokenizer.add_tokens(new_tokens) #返回一个数，表示加入的新词数量，
    #关键步骤，resize_token_embeddings输入的参数是tokenizer的新长度
    print(len(tokenizer))
    model.resize_token_embeddings(len(tokenizer)) 
    tokenizer.save_pretrained(model_checkpoint)