import datasets
from datasets import load_dataset

from model_params import Config
from models import SequenceLabeling
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
import logddd


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

# 获取自己定义的模型 1024 是词表长度 18是标签类别数
multi_calss_model = SequenceLabeling(model, 1024,Config.class_nums,tokenizer)
# 交叉熵损失函数
loss_func_cross_entropy = torch.nn.CrossEntropyLoss()
progress_bar = tqdm(range(num_training_steps))

for epoch in range(Config.num_train_epochs):
    # Training
    multi_calss_model.train()
    for batch in instances:
        score, viterbi,viterbi_score = multi_calss_model(batch)
        # 获取当前的label
        labels = batch["labels"]
        # 找到label中值不为-100的值,也就是真实的label
        label = labels[labels != -100]
        # 将当前的向量转换成onehot
        onehot_label = torch.eye(Config.class_nums)[label]
        # 计算loss
        loss = loss_func_cross_entropy(viterbi_score, onehot_label)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # evaluation
    test_model(model=multi_calss_model,dataloader=instances,tokenizer=tokenizer,epoch=epoch)