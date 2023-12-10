import sys

import joblib
import logddd
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
# from model_fast import SequenceLabeling
from transformers import AutoModelForMaskedLM, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, BertConfig

from model_params import Config
from models import SequenceLabeling

sys.path.append("..")
from data_process.utils import batchify_list, calcu_loss
from predict import test_model
from data_process.data_processing import load_instance_data

import os

pre_train_model_name = ""


def load_model(model_checkpoint):
    """
        根据预训练模型位置加载预训练模型
        @param model_checkpoint 预训练模型位置
    """
    # 获取模型配置
    model_config = BertConfig.from_pretrained(model_checkpoint)
    # 修改配置
    model_config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # 根据当前数据集，往预训练模型中添加标签信息
    tokenizer.add_special_tokens({'additional_special_tokens': Config.special_labels})
    if "bart" in model_checkpoint:
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(model_checkpoint, config=model_config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    multi_class_model = SequenceLabeling(model, 1024, Config.class_nums, tokenizer).to(Config.device)
    return multi_class_model, tokenizer


def train_model(train_data, test_data, model, tokenizer, train_loc, data_size, fold):
    """
        训练模型
        @train_data 训练数据
        @test_data 测试数据
        @model 模型
        @tokenizer tokenizer
        @train_loc ❗️早停时候，模型存储位置，目前没用早停机制了
        @data_size  当前训练集大小
        @fold 当前是五折交叉的第几折
    """
    # optimizer
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    warm_up_ratio = 0.1  # 定义要预热的step
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * Config.num_train_epochs,
    #                                             num_training_steps=Config.num_train_epochs)
    # 获取自己定义的模型 1024 是词表长度 18是标签类别数
    # 交叉熵损失函数
    loss_func_cross_entropy = torch.nn.CrossEntropyLoss()
    # 总的prf值
    total_prf = {
        "recall": 0,
        "f1": 0,
        "precision": 0
    }

    # early_stopping = EarlyStopping(Config.checkpoint_file.format(filename=train_loc), patience=5)
    # 记录train loss 的列表
    loss_list = []
    # 记录test loss的列表
    loss_list_test = []

    start_epoch = 0
    # 创建epoch的进度条
    epochs = trange(start_epoch, Config.num_train_epochs, leave=True, desc="Epoch")
    # 遍历每一个epoch
    for epoch in epochs:
        # Training
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        # 存一个epoch的loss
        total_loss = 0
        logddd.log(len(train_data))
        # train_data是分好batch的
        for batch_index in range(len(train_data)):
            batch = train_data[batch_index]
            _, total_scores, bert_loss = model(batch)

            # 计算loss 这个返回的也是一个batch中，每一条数据的平均loss
            loss = calcu_loss(total_scores, batch, loss_func_cross_entropy)
            # bert的loss 这个是一个batch中，每一条数据的平均loss
            total_loss += loss.item() + bert_loss

            loss.backward()
            # scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item() / Config.batch_size, 5))

        # 模型不会在前10个step收敛，因此前10个step不测试，并且10个step之后隔一个测一次
        if epoch < 10 or epoch % 2 == 1:
            continue
        # 这儿添加的是一个epoch的平均loss
        loss_list.append([total_loss / len(train_data)])
        # tensorboard添加loss
        writer.add_scalar(f'train_loss_{train_loc}', total_loss / len(train_data), epoch)
        # 测试模型 获取prf
        res, test_loss = test_model(model=model, epoch=epoch, writer=writer, loss_func=loss_func_cross_entropy,
                                    dataset=test_data, train_loc=train_loc)
        loss_list_test.append([test_loss])
        # 现在求的不是平均值，而是一次train_model当中的最大值，当前求f1的最大值
        if total_prf["f1"] < res["f1"]:
            total_prf = res

    # 写训练过程中的loss到csv，后面画图
    import csv
    with open(f'{pre_train_model_name}_{data_size}_{fold}_train.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(loss_list)
    with open(f'{pre_train_model_name}_{data_size}_{fold}_test.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(loss_list_test)
    return total_prf


writer = SummaryWriter(Config.log_dir)


def train(model_checkpoint, few_shot_start, data_index):
    """
        模型训练
    """
    # 加载test标准数据
    standard_data_test = joblib.load(Config.test_data_path)
    model_test, tokenizer_test = load_model(model_checkpoint)
    instance_filename = Config.test_data_path.split("/")[-1].replace(".data", "") + ".data"
    if os.path.exists(instance_filename):
        test_data_instances = joblib.load(instance_filename)[:501]
    else:
        # 处理和加载测试数据，并且保存处理之后的结果，下次就不用预处理了
        test_data_instances = load_instance_data(standard_data_test, tokenizer_test, Config, is_train_data=False)
        joblib.dump(test_data_instances, instance_filename)

    del tokenizer_test, model_test
    # 对每一个数量的few-shot进行kfold交叉验证
    for few_shot_idx in range(few_shot_start, len(Config.few_shot)):
        item = Config.few_shot[few_shot_idx]
        logddd.log("当前的训练样本数量为：", item)
        # 加载train数据列表，五折交叉验证，这是一个包含5个列表的列表，其中一个代表一折的数据
        train_data_all = joblib.load(Config.train_data_path.format(item=item))
        # k折交叉验证的prf
        k_fold_prf = {
            "recall": 0,
            "f1": 0,
            "precision": 0
        }
        fold = 1
        for index, standard_data_train in enumerate(train_data_all):
            if index >= Config.kfold:
                break
            train_loc = f"{few_shot_idx}_{index}"
            logddd.log(len(standard_data_train))
            # if Config.resume and index < data_index:
            #     continue
            # 加载model和tokenizer
            model, tokenizer = load_model(model_checkpoint)
            # standard_data_train = split_sentence(standard_data_train)
            # 获取训练数据
            # 将测试数据转为id向量
            train_data_instances = load_instance_data(standard_data_train, tokenizer, Config, is_train_data=True)
            # 划分train数据的batch
            test_data = batchify_list(test_data_instances, batch_size=Config.batch_size)
            train_data = batchify_list(train_data_instances, batch_size=Config.batch_size)

            prf = train_model(train_data, test_data, model, tokenizer, train_loc, len(standard_data_train), fold)
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
    logddd.log(pretrain_model)
    pre_train_model_name = pretrain_model.split("/")[-1]
    train(pretrain_model, 0, 0)
