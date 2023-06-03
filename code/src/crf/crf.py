from typing import List, Dict

from datasets import DatasetDict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm import trange

from model_params import Config
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, BertConfig
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
import logddd
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('log/')

from models import CRFModel
from model_params import Config
import sys
from sklearn import metrics

sys.path.append("..")
from data_process.pos_seg_2_standard import format_data_type_pos_seg


def get_prf(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
        计算prf值
        :param y_true: 真实标签
        :param y_pred: 预测标签
        :return prf值 结构为map key为 recall、f1、precision、accuracy
    """
    res = dict({
        "recall": 0,
        "f1": 0,
        "precision": 0
    })
    res["recall"] = metrics.recall_score(y_true, y_pred, average='macro')
    res["f1"] = metrics.f1_score(y_true, y_pred, average='macro')
    res["precision"] = metrics.precision_score(y_true, y_pred, average='macro')

    return res


def data_reader(filename: str) -> List[str]:
    """
        读取文件，以行为单位返回文件内容
    @param filename: 文件路径
    @return: 以行为单位的内容list
    """
    with open(filename, "r") as f:
        return f.readlines()


def load_data(data_files: str) -> List[List[str]]:
    """
            加载数据 for crf
    @param data_files: 数据文件路径
    @return: 返回训练数据
    """
    datas = data_reader(data_files)
    # 转换为标准数据
    standard_data = format_data_type_pos_seg(datas)
    return standard_data


def load_instance_data(standard_data: List[List[str]], tokenizer, Config, is_train_data: bool):
    """
      加载训练用的数据集
      :param standard_data: 标准格式的数据
      :param tokenizer:tokenizer对象
      :param Config: 模型配置类
      :param is_train_data: 是否是加载train数据，train 和test的数据加载有一定的偏差
    """
    # 每一条数据转换成输入模型内的格式
    instance_data = []
    for data in standard_data:
        sequence = data[0].strip().split("/")
        labels = data[1].strip().replace("\n", "").split("/")

        # 手动转为id列表
        input_ids = []
        attention_mask = []
        label_ids = []
        for i in range(Config.sentence_max_len):
            if i < len(sequence):
                input_ids.append(tokenizer.convert_tokens_to_ids(sequence[i]))
                attention_mask.append(1)
                label_ids.append(tokenizer.convert_tokens_to_ids(labels[i]))
            else:
                input_ids.append(0)
                attention_mask.append(0)
                label_ids.append(-100)
        result = {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
            "labels": [label_ids]
        }
        instance_data.append(result)

    return instance_data


def batchify_list(data, batch_size):
    """
        将输入数据按照batch_size进行划分
        @param data:输入数据
        @param batch_size: batch_size
        @return batch_data
    """
    batched_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batched_data.append(batch)

    return batched_data


def load_model():
    # 加载模型名字
    model_checkpoint = Config.model_checkpoint
    # 获取模型配置
    # model_config = BertConfig.from_pretrained(model_checkpoint)
    # 修改配置
    # model_config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': ["[PLB]", "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
                                                                "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
                                                                "VA",
                                                                "VE"]})
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    multi_class_model = CRFModel(model, Config.class_nums, tokenizer).to(Config.device)
    return multi_class_model, tokenizer


def test_model(model, epoch, writer, dataset):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        # 总的预测出来的标签
        total_y_pre = []
        total_y_true = []
        for batch in test_data:
            datas = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            for data in batch:
                for k, v in data.items():
                    datas[k].extend(v)
            # 获取所有的真实label

            for sentence in datas["labels"]:
                for item in sentence:
                    if item != -100:
                        total_y_true.append(item)
            # 将数据转为tensor
            batch_data = {
                k: torch.tensor(v).to(Config.device)
                for k, v in datas.items()
            }

            loss, paths = model(batch_data)
            # 获取预测的label
            for path in paths:
                total_y_pre.extend(path)

            total_loss += loss.item()

        writer.add_scalar('test_loss', total_loss / len(test_data), epoch)
        logddd.log(len(total_y_pre))
        logddd.log(len(total_y_true))
        report = classification_report(total_y_true, total_y_pre)
        print()
        print(report)
        print()
        res = get_prf(y_true=total_y_true, y_pred=total_y_pre)
        return res


def train_model(train_data, test_data, model, tokenizer):
    """
        训练模型
    """
    # optimizer
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

    # 获取自己定义的模型 1024 是词表长度 18是标签类别数

    # 交叉熵损失函数
    loss_func_cross_entropy = torch.nn.CrossEntropyLoss()
    # 创建epoch的进度条
    epochs = trange(Config.num_train_epochs, leave=True, desc="Epoch")
    # 总的prf值
    total_prf = {
        "recall": 0,
        "f1": 0,
        "precision": 0
    }
    for epoch in epochs:
        # Training
        model.train()
        total_loss = 0
        for batch_index in tqdm(range(len(train_data)), total=len(train_data), desc="Batchs"):
            batch = train_data[batch_index]
            # 模型计算
            datas = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            for data in batch:
                for k, v in data.items():
                    datas[k].extend(v)

            batch_data = {
                k: torch.tensor(v).to(Config.device)
                for k, v in datas.items()
            }

            loss, _ = model(batch_data)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item() / Config.batch_size, 5))
            loss.cpu()
            del loss

        writer.add_scalar('train_loss', total_loss / len(train_data), epoch)
        res = test_model(model=model, epoch=epoch, writer=writer, dataset=test_data)
        # 现在求的不是平均值，而是一次train_model当中的最大值，当前求f1的最大值
        if total_prf["f1"] < res["f1"]:
            total_prf = res


    del model
    return total_prf


# 加载标准数据
standard_data = load_data(Config.dataset_path)
# 创建一个空的y数组，真实y标签在instances里面
y_none_use = [0] * len(standard_data)
# 创建K折交叉验证迭代器
kfold = StratifiedKFold(n_splits=Config.kfold, shuffle=True)
# 加载模型，获取tokenizer

k_fold_prf = {
    "recall": 0,
    "f1": 0,
    "precision": 0
}
for train, val in kfold.split(standard_data, y_none_use):
    model, tokenizer = load_model()
    # 获取train的标准数据和test的标准数据
    train_standard_data = [standard_data[x] for x in train]
    test_standard_data = [standard_data[x] for x in val]
    # 将标准数据转换为id向量
    train_data_instances = load_instance_data(train_standard_data, tokenizer, Config, is_train_data=True)
    test_data_instances = load_instance_data(test_standard_data, tokenizer, Config, is_train_data=False)
    # 划分train数据的batch
    train_data = batchify_list(train_data_instances, batch_size=Config.batch_size)
    test_data = batchify_list(test_data_instances, batch_size=Config.batch_size)
    prf = train_model(train_data, test_data, model, tokenizer)
    for k, v in prf.items():
        k_fold_prf[k] += v
    logddd.log("epoch测试")
    exit(0)
    del model, tokenizer

avg_prf = {
    k: v / Config.kfold
    for k, v in k_fold_prf.items()
}
logddd.log(avg_prf)

# if __name__ == '__main__':
#     model, tokenizer = load_model()
#     standard_data = load_data("/home/dlf/prompt/code/data/jw/after_pos_seg.txt")
#     instances = load_instance_data(standard_data, tokenizer, Config, True)
