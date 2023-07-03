from typing import List, Dict

import joblib
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

from models import BiLSTMCRFModel
from model_params import Config
import sys
from sklearn import metrics

sys.path.append("..")
from data_process.pos_seg_2_standard import format_data_type_pos_seg

from utils import EarlyStopping

writer = SummaryWriter('log/')


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

    # 获取模型配置
    # model_config = BertConfig.from_pretrained(model_checkpoint)
    # 修改配置
    # model_config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': Config.special_labels})
    if "bart" in model_checkpoint:
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    multi_class_model = BiLSTMCRFModel(model, Config.class_nums, tokenizer).to(Config.device)
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

            loss, paths = model(batch_data)
            # 获取预测的label
            for path in paths:
                total_y_pre.extend(path)

            total_loss += loss.item()

        writer.add_scalar('test_loss', total_loss / len(test_data), epoch)
        report = classification_report(total_y_true, total_y_pre)
        print()
        print(report)
        print()
        res = get_prf(y_true=total_y_true, y_pred=total_y_pre)
        return res, total_loss / len(test_data)


import os


def load_start_epoch(model, optimizer):
    """
        加载检查点
    """
    start_epoch = -1
    if Config.resume and os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth")  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        os.rename("checkpoint.pth", "checkpoint_old.pth")
    return start_epoch


def save_checkpoint(model, optimizer, epoch):
    """
        保存检查点
    """
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, 'checkpoint.pth')


def train_model(train_data, test_data, model, tokenizer):
    """
        训练模型
    """
    # optimizer
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

    # 获取自己定义的模型 1024 是词表长度 18是标签类别数

    # 交叉熵损失函数
    loss_func_cross_entropy = torch.nn.CrossEntropyLoss()
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
    early_stopping = EarlyStopping("")
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

            loss, _ = model(batch_data)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item() / Config.batch_size, 5))
            loss.cpu()
            del loss
            # # 如果不是最后一个epoch，那就保存检查点
            # if epoch != len(epochs) - 1:
            #     save_checkpoint(model, optimizer, epoch)

        writer.add_scalar('train_loss', total_loss / len(train_data), epoch)
        res, test_loss = test_model(model=model, epoch=epoch, writer=writer, test_data=test_data)
        # 现在求的不是平均值，而是一次train_model当中的最大值，当前求f1的最大值
        if total_prf["f1"] < res["f1"]:
            total_prf = res

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            logddd.log("early stop")
            break

    del model
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

            prf = train_model(train_data, test_data, model, tokenizer)
            logddd.log("当前fold为：", fold)
            fold += 1
            logddd.log("当前的train的最优值")
            logddd.log(prf)
            for k, v in prf.items():
                k_fold_prf[k] += v

            check_point_outer = {
                "few_shot_idx": few_shot_idx,
                "train_data_idx": index,
                "model": model_checkpoint
            }

            # if index != len(train_data) - 1:
            #     joblib.dump(check_point_outer, "checkpoint_outer.data")
            del model, tokenizer

        avg_prf = {
            k: v / Config.kfold
            for k, v in k_fold_prf.items()
        }
        logddd.log(avg_prf)
        prf = f"当前train数量为:{item}"
        logddd.log(prf)


for pretrain_model in Config.pretrain_models:
    prf = pretrain_model
    logddd.log(prf)
    # if os.path.exists("checkpoint_outer.data") and Config.resume:
    #     check_point_outer = joblib.load("check_point_outer")
    #     os.rename("checkpoint_outer.data", "checkpoint_outer_older.data")
    #     if check_point_outer['model'] == pretrain_model:
    #         train(pretrain_model, check_point_outer["few_shot_idx"], check_point_outer["train_data_idx"])
    #         continue
    if "bert_large" in pretrain_model:
        train(pretrain_model, 1, 0)
    else:
        train(pretrain_model, 0, 0)
