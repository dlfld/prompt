from typing import List

import torch
import tqdm
from sklearn.metrics import classification_report
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

from data_process.utils import calcu_loss
from utils import get_prf


def link_predict(model, epoch, writer, loss_func, test_data, train_loc):
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
    for batch in tqdm.tqdm(test_data, desc="test"):
        for index, datas in enumerate(batch):
            # 取出真实的label
            labels = datas["labels"]
            # logddd.log(labels)
            # 将所有的ytrue放到一起
            for label_idx in range(len(labels)):
                item = labels[label_idx]
                for y in item:
                    if y != -100:
                        total_y_true.append(y.item())

        seq_predict_labels, scores, bert_loss = model(batch)
        # 将所有的y pre放到一起
        for path in seq_predict_labels:
            total_y_pre.extend([x + 1 for x in path])

        loss = calcu_loss(scores, batch, loss_func)

        total_loss += loss.item() + bert_loss
        del loss, bert_loss, scores

    writer.add_scalar(f'test_loss_{train_loc}', total_loss / len(test_data), epoch)
    report = classification_report(total_y_true, total_y_pre)
    print()
    print(report)
    print()
    res = get_prf(y_true=total_y_true, y_pred=total_y_pre)

    return res, total_loss / len(test_data)


def save_predict_file(file_path: str, content: List[str]):
    """
    存储预测结果文件
    :param file_path: 结果文件
    :param content: 内容
   """
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(content)


def test_model(model, epoch, writer, loss_func, dataset, train_loc):
    """
        加载验证集 链式预测
        :param model: 模型
        :param epoch: 当前的轮数
        :param writer:参数
    """
    model.eval()
    with torch.no_grad():
        # 链式调用预测
        res, test_loss = link_predict(model, epoch, writer, loss_func, dataset, train_loc)

    model.train()
    return res, test_loss


if __name__ == '__main__':
    model_checkpoint = "/home/dlf/crf/code/model/bert_large_chinese"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    link_predict(model, tokenizer, 1)
