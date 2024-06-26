import sys
from typing import List, Dict

import joblib
import logddd
import torch
from sklearn import metrics
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from tqdm.auto import tqdm

from model_params import Config
from models import BiLSTMCRFModel

sys.path.append("..")


writer = SummaryWriter(Config.log_dir)
pre_train_model_name = ""


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
    res["recall"] = metrics.recall_score(y_true, y_pred, average='weighted')
    res["f1"] = metrics.f1_score(y_true, y_pred, average='weighted')
    res["precision"] = metrics.precision_score(y_true, y_pred, average='weighted')

    return res


def data_reader(filename: str) -> List[str]:
    """
        读取文件，以行为单位返回文件内容
    @param filename: 文件路径
    @return: 以行为单位的内容list
    """
    with open(filename, "r") as f:
        return f.readlines()

def format_data_type_pos_seg(datas: List[str]) -> List[List[str]]:
    """
        对蒋文的数据集进行标准格式化处理
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


def load_model(model_checkpoint):
    # 加载模型名字
    from transformers import AutoTokenizer, AutoConfig
    from transformers import BertModel, BartModel
    # 获取模型配置
    # model_config = BertConfig.from_pretrained(model_checkpoint)
    # 修改配置
    # model_config.output_hidden_states = True
    model_config = AutoConfig.from_pretrained(model_checkpoint)
    # 修改配置
    model_config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': Config.special_labels})
    if "bart" in model_checkpoint:
        model = BartModel(model_config)
    else:
        model = BertModel(model_config)
    model.resize_token_embeddings(len(tokenizer))
    multi_class_model = BiLSTMCRFModel(model, Config.class_nums, tokenizer, model_config, model_checkpoint).to(
        Config.device)
    return multi_class_model, tokenizer


def test_model(model, epoch, writer, test_data):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        # 总的预测出来的标签
        total_y_pre = []
        total_y_true = []
        for batch in tqdm(test_data, desc="test"):

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

            logits, loss, paths = model(batch_data)
            # # 获取预测的label
            # for path in paths:
            total_y_pre.extend(paths)

            total_loss += loss.item()

        writer.add_scalar('test_loss', total_loss / len(test_data), epoch)

        report = classification_report(total_y_true, total_y_pre)
        print()
        print(report)
        print()
        res = get_prf(y_true=total_y_true, y_pred=total_y_pre)
        return res, total_loss / len(test_data)


import os


def train_model(train_data, test_data, model, tokenizer, data_size, fold):
    """
        训练模型
    """
    # optimizer
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

    # 获取自己定义的模型 1024 是词表长度 18是标签类别数

    # 加载开始epoch
    # start_epoch = load_start_epoch(model, optimizer)
    start_epoch = -1
    # 创建epoch的进度条
    epochs = trange(start_epoch + 1, Config.num_train_epochs, leave=True, desc="Epoch")
    # 总的prf值
    total_prf = {
        "recall": 0,
        "f1": 0,
        "precision": 0
    }

    loss_list = []
    loss_list_test = []
    for epoch in epochs:
        # Training
        model.train()
        total_loss = 0
        for batch_index in range(len(train_data)):
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

            _, loss, _ = model(batch_data)

            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item() / Config.batch_size, 5))


        writer.add_scalar('train_loss', total_loss / len(train_data), epoch)
        loss_list.append([total_loss / len(train_data)])
        res, test_loss = test_model(model=model, epoch=epoch, writer=writer, test_data=test_data)
        loss_list_test.append([test_loss])
        # 现在求的不是平均值，而是一次train_model当中的最大值，当前求f1的最大值
        if total_prf["f1"] < res["f1"]:
            total_prf = res


    # del model
    import csv
    with open(f'{pre_train_model_name}_{data_size}_{fold}_train.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(loss_list)

    with open(f'{pre_train_model_name}_{data_size}_{fold}_test.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(loss_list_test)
    return total_prf


def train(model_checkpoint, few_shot_start, data_index):
    # 加载test标准数据
    standard_data_test = joblib.load(Config.test_data_path)
    # 对每一个数量的few-shot进行kfold交叉验证
    for few_shot_idx in range(few_shot_start, len(Config.few_shot)):
        item = Config.few_shot[few_shot_idx]
        logddd.log("当前的训练样本数量为：", item)
        # 加载train数据列表
        train_data = joblib.load(Config.train_data_path.format(item=item))

        # k折交叉验证的prf
        k_fold_prf = {
            "recall": 0,
            "f1": 0,
            "precision": 0
        }
        fold = 1
        # for index in range(Config.kfold):
        for index, standard_data_train in enumerate(train_data):
            if index < data_index:
                continue
            if index >= Config.kfold:
                break
            # 加载model和tokenizer
            model, tokenizer = load_model(model_checkpoint)

            # 获取训练数据
            # standard_data_train = train_data[index]
            # 将测试数据转为id向量
            test_data_instances = load_instance_data(standard_data_test, tokenizer, Config, is_train_data=False)
            train_data_instances = load_instance_data(standard_data_train, tokenizer, Config, is_train_data=True)

            # 划分train数据的batch
            test_data = batchify_list(test_data_instances, batch_size=Config.batch_size)
            train_data = batchify_list(train_data_instances, batch_size=Config.batch_size)

            prf = train_model(train_data, test_data, model, tokenizer, len(standard_data_train), fold)
            logddd.log("当前fold为：", fold)
            fold += 1
            logddd.log("当前的train的最优值")
            logddd.log(prf)
            for k, v in prf.items():
                k_fold_prf[k] += v

            del model, tokenizer

        avg_prf = {
            k: v / Config.kfold
            for k, v in k_fold_prf.items()
        }
        logddd.log(avg_prf)
        prf = f"当前train数量为:{item}"
        logddd.log(prf)


for pretrain_model in Config.pretrain_models:
    logddd.log(pretrain_model)
    pre_train_model_name = pretrain_model.split("/")[-1]

    train(pretrain_model, 0, 0)

