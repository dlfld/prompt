import copy
from typing import List

import math
from sklearn.metrics import classification_report
from utils import get_prf
from model_params import Config
from data_process.data_processing import generate_prompt, load_data, load_instance_data
from data_process.utils import data_reader
from data_process.utils import batchify_list, calcu_loss
from data_process.pos_seg_2_standard import format_data_type_pos_seg

import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import logddd
from data_process.utils import batchify_list


def link_predict(model, epoch, writer, loss_func, test_data):
    """
        使用模型进行链式的预测
        :param model:   模型
        :param tokenizer: tokenizer对象
        :param epoch: epoch
        
    """
    total_prompt_nums = 0
    total_loss = 0
    # 总的预测出来的标签
    total_y_pre = []
    total_y_true = []
    for batch in test_data:
        # 模型计算
        # datas = {
        #     "input_ids":[],
        #     "attention_mask":[],
        #     "labels":[]
        # }
        # for data in batch:
        #     for k,v in data.items():
        #         datas[k].extend(v.tolist())
        for index, datas in enumerate(batch):
            # 取出真实的label
            labels = datas["labels"]
            # logddd.log(labels)
            # 将所有的ytrue放到一起
            for label_idx in range(len(labels)):
                item = labels[label_idx]
                for y in item:
                    if y != -100:
                        total_y_true.append(y)

        seq_predict_labels, scores, bert_loss = model(batch)
        # 将所有的y pre放到一起
        for path in seq_predict_labels:
            total_y_pre.extend([x-1 for x in path])

        loss = calcu_loss(scores, batch, loss_func)
        loss += bert_loss
        total_loss += loss.item()
        del loss


    writer.add_scalar('test_loss', total_loss / len(test_data), epoch)
    report = classification_report(total_y_true, total_y_pre)
    print()
    print(report)
    print()
    res = get_prf(y_true=total_y_true, y_pred=total_y_pre)
    return res


def save_predict_file(file_path: str, content: List[str]):
    """
    存储预测结果文件
    :param file_path: 结果文件
    :param content: 内容
   """
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(content)


def test_model(model, epoch, writer, loss_func, dataset):
    """
        加载验证集 链式预测
        :param model: 模型
        :param epoch: 当前的轮数
        :param writer:参数
    """
    model.eval()

    with torch.no_grad():
        # 链式调用预测
        res = link_predict(model, epoch, writer, loss_func, dataset)
        return res


if __name__ == '__main__':
    model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    link_predict(model, tokenizer, 1)
