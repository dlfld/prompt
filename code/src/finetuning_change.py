import datasets
from datasets import load_dataset
from models import MultiClass
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer,BertConfig
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
from data_process.data_processing import load_data,load_instance_data
class Config(object):
    """
        配置类，保存配置文件
    """
    # 训练集位置
    train_dataset_path = "/home/dlf/prompt/code/data/jw/short_data_train.txt"
    # 测试集位置
    test_dataset_path = "/home/dlf/prompt/code/data/jw/short_data_train.txt"
    # prompt dataset
    # train_dataset_path = "/home/dlf/prompt/dataset.csv"
    # 预训练模型的位置
    model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
    # 训练集大小
    train_size = 60
    # batch_size
    batch_size = 16
    # 学习率
    learning_rate = 2e-5
    # epoch数
    num_train_epochs = 3
    # 句子的最大补齐长度
    sentence_max_len = 128
    # 结果文件存储位置
    predict_res_file = "/home/dlf/prompt/code/res_files/short_data_res_{}.txt"


# 加载标准数据
standard_data = load_data(Config.train_dataset_path)
# 加载模型名字
model_checkpoint = Config.model_checkpoint
# 获取模型配置
model_config = BertConfig.from_pretrained(model_checkpoint)
# 修改配置
model_config.output_hidden_states = True
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint,config = model_config)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# 加载训练数据
instances = load_instance_data(standard_data,tokenizer,Config)


model_name = model_checkpoint.split("/")[-1]

logging_steps = len(instances)
batch_size = 1
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

# # 获取自己定义的模型 21128 是词表长度 18是标签类别数
multi_calss_model = MultiClass(model, 1024,18)
criterion = torch.nn.BCEWithLogitsLoss()
progress_bar = tqdm(range(num_training_steps))

for epoch in range(Config.num_train_epochs):
    # Training
    model.train()
    for batch in instances:
        # print(batch)
        # print(type(batch))    <class 'dict'>
        # print(batch.keys())  dict_keys(['input_ids', 'attention_mask', 'labels'])
        # outputs = model(**batch)
        multi_calss_model(batch)
        # exit(0)
        # loss = outputs.loss
        # logits = outputs.logits
        #
        # loss.backward()
        # optimizer.step()
        # lr_scheduler.step()
        # optimizer.zero_grad()
        # progress_bar.update(1)
    # evaluation
    test_model(model=model,dataloader=instances,tokenizer=tokenizer,epoch=epoch,Config=Config)