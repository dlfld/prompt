import datasets
from datasets import load_dataset

from model_params import Config
from models import SequenceLabeling
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, BertConfig
import copy
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import math
from predict import link_predict, test_model
from data_process.data_processing import load_data, load_instance_data
import logddd


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


def calcu_loss(bert_loss, total_scores, batch):
    """
        计算loss
        @param bert_loss: 一个batch的路径
        @param total_scores： 一个batch计算过程中的score矩阵
        @param 一个batch的数据
        @return loss
    """
    # total_loss = 0
    # for data_idx in range(len(batch)):
    #     data = batch[data_idx]
    #     labels = []
    #     for idx in range(len(data)):
    #         item = data[idx]["labels"]
    #         label = item[item != -100]
    #         # logddd.log(label)
    #         onehot_label = torch.eye(Config.class_nums)[label]
    #         # logddd.log(onehot_label)
    #         labels.append(onehot_label.tolist())
    #
    #     onehot_predict = torch.eye(Config.class_nums)[total_path[data_idx]]
    #     cur_labels = torch.tensor(labels)
    #     cur_labels = torch.squeeze(cur_labels, dim = 1)
    #     cur_predict = torch.tensor(onehot_predict,requires_grad=True)
    #     # logddd.log(cur_predict.shape)
    #     # logddd.log(cur_labels.shape)
    #     cur_loss = loss_func_cross_entropy(cur_predict, cur_labels)
    #     total_loss += cur_loss
    # return total_loss / len(batch)
    # =============================
    labels = batch["labels"]
    onehot_labels = []
    for label_idx in range(len(labels)):
        item = labels[label_idx]
        label = [x-1 for x in item if x != -100]
        onehot_label = torch.eye(Config.class_nums)[label]
        onehot_labels.append(onehot_label.tolist())

    onehot_labels = torch.tensor(onehot_labels).to(Config.device)

    onehot_labels = torch.squeeze(onehot_labels,dim=1)
    total_scores = torch.tensor(total_scores,requires_grad=True).to(Config.device)
    cur_loss = loss_func_cross_entropy(total_scores, onehot_labels)
    return cur_loss


    # =============================
    total_loss = 0

    for data_idx in range(len(batch)):
        data = batch[data_idx]
        labels = []
        for idx in range(len(data)):
            item = data[idx]["labels"]
            label = item[item != -100]
            # 对应到0-17的onehot上
            label[0] -= 1
            onehot_label = torch.eye(Config.class_nums)[label]
            labels.append(onehot_label.tolist())

        cur_score = torch.tensor(total_scores[data_idx], requires_grad=True) .to(Config.device)
        cur_labels = torch.tensor(labels) .to(Config.device)
        cur_labels = torch.squeeze(cur_labels, dim=1)
        # cur_score = torch.squeeze(cur_score,dim=1)
        cur_loss = loss_func_cross_entropy(cur_score, cur_labels)
        total_loss += cur_loss
    return total_loss / len(batch)


# 加载标准数据
standard_data = load_data(Config.train_dataset_path)
# 加载模型名字
model_checkpoint = Config.model_checkpoint
# 获取模型配置
model_config = BertConfig.from_pretrained(model_checkpoint)
# 修改配置
model_config.output_hidden_states = True
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({'additional_special_tokens': ["[PLB]","NR", "NN", "AD", "PN", "OD", "CC", "DEG", "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC", "VA",
              "VE"]})
# new_tokens = ["NR", "NN", "AD", "PN", "OD", "CC", "DEG", "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC", "VA",
#               "VE"]
# tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
# 加载训练数据
instances = load_instance_data(standard_data, tokenizer, Config)
train_data = batchify_list(instances, batch_size=2)

model_name = model_checkpoint.split("/")[-1]

logging_steps = len(instances)
batch_size = Config.batch_size
training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=Config.learning_rate,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_update_steps_per_epoch = len(instances)
num_training_steps = Config.num_train_epochs * num_update_steps_per_epoch

# 根据epoch的大小来调整学习率
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 添加loss 计算

# 获取自己定义的模型 1024 是词表长度 18是标签类别数
multi_class_model = SequenceLabeling(model, 1024, Config.class_nums, tokenizer).to(Config.device)
# 交叉熵损失函数
loss_func_cross_entropy = torch.nn.CrossEntropyLoss()
progress_bar = tqdm(range(num_update_steps_per_epoch))

for epoch in range(Config.num_train_epochs):
    # Training
    multi_class_model.train()
    Config.model_train = True

    for batch in train_data:
        # 模型计算
        datas = {
            "input_ids":[],
            "attention_mask":[],
            "labels":[]
        }
        for data in batch:
            for k,v in data.items():
                datas[k].extend(v.tolist())
        # for data in datas["input_ids"]:s
        _, total_scores,bert_loss = multi_class_model(datas)

        # 计算loss
        loss = calcu_loss(bert_loss, total_scores, datas)
        # logddd.log(loss,bert_loss)
        # print(loss,bert_loss)
        loss += bert_loss
        # logddd.log(loss)
        loss.backward()
        # logddd.log(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    # evaluation
    multi_class_model.eval()
    Config.model_train = False
    test_model(model=multi_class_model, dataloader=instances, tokenizer=tokenizer, epoch=epoch)
