import sys

import joblib
import logddd
import torch
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
# from model_fast import SequenceLabeling
from transformers import AutoModelForMaskedLM, get_linear_schedule_with_warmup, AdamW
from transformers import AutoTokenizer, BertConfig

from model_params import Config
from models import SequenceLabeling

sys.path.append("..")
from data_process.utils import batchify_list, calcu_loss
from predict import test_model
from data_process.data_processing import load_instance_data
from utils import EarlyStopping

import os

pre_train_model_name = ""


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
    # model_config.hidden_size = 768
    # model_config.output_hidden_states = len(Config.special_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
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
        "precision": 0,
        "acc": 0
    }

    early_stopping = EarlyStopping(Config.checkpoint_file.format(filename=train_loc), patience=5)
    loss_list = []
    loss_list_test = []

    for epoch in epochs:
        # Training
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        # 存一个epoch的loss
        total_loss = 0
        logddd.log(len(train_data))

        for batch_index in range(len(train_data)):
            batch = train_data[batch_index]
            _, total_scores, bert_loss = model(batch)
            # 计算loss 这个返回的也是一个batch中，每一条数据的平均loss
            loss = calcu_loss(total_scores, batch, loss_func_cross_entropy)
            # bert的loss 这个是一个batch中，每一条数据的平均loss
            total_loss += loss.item() + bert_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item() / Config.batch_size, 5))

        if epoch < 10 or epoch % 2 == 0:
            continue
        # 这儿添加的是一个epoch的平均loss
        loss_list.append([total_loss / len(train_data)])
        writer.add_scalar(f'train_loss_{train_loc}', total_loss / len(train_data), epoch)
        res, test_loss = test_model(model=model, epoch=epoch, writer=writer, loss_func=loss_func_cross_entropy,
                                    dataset=test_data, train_loc=train_loc)
        loss_list_test.append([test_loss])

        # 现在求的不是平均值，而是一次train_model当中的最大值，当前求f1的最大值
        if total_prf["f1"] < res["f1"]:
            total_prf = res

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            logddd.log("early stop")
        # break

    import csv
    with open(f'{pre_train_model_name}_{data_size}_{fold}_train.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(loss_list)
    with open(f'{pre_train_model_name}_{data_size}_{fold}_test.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(loss_list_test)
    return total_prf


writer = SummaryWriter(Config.log_dir)


def split_sentence(standard_datas):
    """
    主要是因为句子太长之后batch_size设置为1也会炸显存，
        因此，这个地方以逗号分隔句子，在分隔完成之后发现也会炸显存，
        逐个排除之后发现24G的显存支持长度为8的句子。所以按照8个词为单位和逗号进行分隔

        有两种情况，1. 按照8个词分隔，
                  2. 按照逗号和8个词分隔
    根据逗号划分句子
    """
    res_data = []
    for data in standard_datas:
        sentence = data[0].split("/")
        labels = data[1].split("/")
        item = [[], []]
        for i in range(len(sentence)):

            if len(item[0]) < 10:
                # if sentence[i] != '，' and len(item[0]) < Config.pre_n:
                item[0].append(sentence[i])
                item[1].append(labels[i])
            else:
                res_data.append(["/".join(item[0]), "/".join(item[1])])
                item = [[], []]
                # 重制之后再添加单词信息
                item[0].append(sentence[i])
                item[1].append(labels[i])

        res_data.append(["/".join(item[0]), "/".join(item[1])])

    return res_data


def train(model_checkpoint, few_shot_start, data_index):
    # 加载test标准数据
    standard_data_test = joblib.load(Config.test_data_path)
    model_test, tokenizer_test = load_model(model_checkpoint)
    # logddd.log(tokenizer_test.convert_ids_to_tokens([99]))
    # exit(0)
    # standard_data_test = split_sentence(standard_data_test)
    instance_filename = Config.test_data_path.split("/")[-1].replace(".data", "") + ".data"
    if os.path.exists(instance_filename):
        # 加载测试数据集
        test_data_instances = joblib.load(instance_filename)
    else:
        test_data_instances = load_instance_data(standard_data_test, tokenizer_test, Config, is_train_data=False)
        joblib.dump(test_data_instances, instance_filename)
    test_data_instances = test_data_instances[:300]
    # test_data_instances = test_data_instances[:40]
    # logddd.log(tokenizer_test.convert_ids_to_tokens(test_data_instances[0]["input_ids"][0]))
    # logddd.log(tokenizer_test.convert_tokens_to_ids(test_data_instances[0]["labels"][0]))
    # exit(0)
    # test_data_instances = test_data_instances[:50]
    # test_data_instances = joblib.load("/home/dlf/crf/code/src/crf/bert_test_data_instance.data")
    # logddd.log(test_data_instances)
    del tokenizer_test, model_test
    # 对每一个数量的few-shot进行kfold交叉验证
    for few_shot_idx in range(few_shot_start, len(Config.few_shot)):
        item = Config.few_shot[few_shot_idx]
        logddd.log("当前的训练样本数量为：", item)
        # 加载train数据列表
        train_data_all = joblib.load(Config.train_data_path.format(item=item))

        # k折交叉验证的prf
        k_fold_prf = {
            "recall": 0,
            "f1": 0,
            "precision": 0,
            "acc": 0,
        }
        fold = data_index + 1
        # for index in range(Config.kfold):
        for index, standard_data_train in enumerate(train_data_all):
            if index >= Config.kfold:
                break
            if index < data_index:
                continue
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

            # prf = train_model(train_data, test_data, model, tokenizer)
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
    prf = pretrain_model
    logddd.log(prf)

    pre_train_model_name = pretrain_model.split("/")[-1]

    train(pretrain_model, 0, 0)
