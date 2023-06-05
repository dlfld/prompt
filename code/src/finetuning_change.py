from sklearn.model_selection import StratifiedKFold
from tqdm import trange

from model_params import Config
from models import SequenceLabeling
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, BertConfig
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from data_process.utils import batchify_list, calcu_loss
from predict import test_model
from data_process.data_processing import load_data, load_instance_data
import logddd
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('log/')


def load_model():
    # 加载模型名字
    model_checkpoint = Config.model_checkpoint
    # 获取模型配置
    model_config = BertConfig.from_pretrained(model_checkpoint)
    # 修改配置
    model_config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': ["[PLB]", "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
                                                                "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
                                                                "VA",
                                                                "VE"]})
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
                    datas[k].extend(v.tolist())
            _, total_scores, bert_loss = model(datas)

            # 计算loss
            loss = calcu_loss(total_scores, datas, loss_func_cross_entropy)
            loss += bert_loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item() / Config.batch_size, 5))
            loss.cpu()
            bert_loss.cpu()
            del loss
            del bert_loss

        writer.add_scalar('train_loss', total_loss / len(train_data), epoch)
        res = test_model(model=model, epoch=epoch, writer=writer, loss_func=loss_func_cross_entropy, dataset=test_data)
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
    logddd.log("当前的train的最优值", prf)
    for k, v in prf.items():
        k_fold_prf[k] += v

    del model, tokenizer

avg_prf = {
    k: v / Config.kfold
    for k, v in k_fold_prf.items()
}
logddd.log(avg_prf)
