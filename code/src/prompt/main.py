import joblib
from sklearn.model_selection import StratifiedKFold
from tqdm import trange
from model_params import Config
from models import SequenceLabeling
# from model_fast import SequenceLabeling
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, BertConfig
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
import logddd
from torch.utils.tensorboard import SummaryWriter
import sys
import copy

sys.path.append("..")
from data_process.utils import batchify_list, calcu_loss
from predict import test_model
from data_process.data_processing import load_instance_data
from utils import EarlyStopping

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


def load_model(model_checkpoint):
    # 获取模型配置
    model_config = BertConfig.from_pretrained(model_checkpoint)
    # 修改配置
    model_config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': Config.special_labels})
    if "bart" in model_checkpoint:
        from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
        model = BartForConditionalGeneration.from_pretrained(model_checkpoint, config=model_config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    multi_class_model = SequenceLabeling(model, 1024, Config.class_nums, tokenizer).to(Config.device)
    return multi_class_model, tokenizer


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
    start_epoch = load_start_epoch(model, optimizer)
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
            _, total_scores, bert_loss = model(batch)
            # 计算loss
            loss = calcu_loss(total_scores, batch, loss_func_cross_entropy)
            loss += bert_loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item() / Config.batch_size, 5))
            # del loss
            # del bert_loss
            # 如果不是最后一个epoch，那就保存检查点
            if epoch != len(epochs) - 1:
                save_checkpoint(model, optimizer, epoch)

        writer.add_scalar('train_loss', total_loss / len(train_data), epoch)
        res, test_loss = test_model(model=model, epoch=epoch, writer=writer, loss_func=loss_func_cross_entropy,
                                    dataset=test_data)

        # 现在求的不是平均值，而是一次train_model当中的最大值，当前求f1的最大值
        if total_prf["f1"] < res["f1"]:
            total_prf = res

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            logddd.log("early stop")
            break

    return total_prf


writer = SummaryWriter('log/')


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

            if index != len(train_data) - 1:
                joblib.dump(check_point_outer, "checkpoint_outer.data")
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
    if os.path.exists("checkpoint_outer.data") and Config.resume:
        check_point_outer = joblib.load("check_point_outer")
        os.rename("checkpoint_outer.data", "checkpoint_outer_older.data")
        if check_point_outer['model'] == pretrain_model:
            train(pretrain_model, check_point_outer["few_shot_idx"], check_point_outer["train_data_idx"])
            continue
    train(pretrain_model, 0, 0)
