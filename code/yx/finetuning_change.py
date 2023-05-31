import datasets
from datasets import load_dataset
from tqdm import trange

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
from data_process.utils import batchify_list,calcu_loss
import math
from predict import link_predict, test_model
from data_process.data_processing import load_data, load_instance_data
import logddd
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log/')
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
# 加载标准数据
standard_data = load_data(Config.train_dataset_path)
instances = load_instance_data(standard_data, tokenizer, Config,is_train_data=True)
train_data = batchify_list(instances, batch_size=Config.batch_size)

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

optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

num_update_steps_per_epoch = len(instances)
num_training_steps = Config.num_train_epochs * num_update_steps_per_epoch

# 根据epoch的大小来调整学习率
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )

# 获取自己定义的模型 1024 是词表长度 18是标签类别数
multi_class_model = SequenceLabeling(model, 1024, Config.class_nums, tokenizer).to(Config.device)
# 交叉熵损失函数
loss_func_cross_entropy = torch.nn.CrossEntropyLoss()
# progress_bar = tqdm(range(num_update_steps_per_epoch),desc="epoch:")
batch_step = 0
epochs = trange(Config.num_train_epochs, leave=True, desc="Epoch")
for epoch in epochs:
    # Training
    multi_class_model.train()
    total_loss = 0
    for batch_index in tqdm(range(len(train_data)),total=len(train_data),desc="Batchs"):
        batch = train_data[batch_index]
        # 模型计算
        datas = {
            "input_ids":[],
            "attention_mask":[],
            "labels":[]
        }
        for data in batch:
            for k,v in data.items():
                datas[k].extend(v.tolist())
        _, total_scores,bert_loss = multi_class_model(datas)

        # 计算loss
        loss = calcu_loss( total_scores, datas,loss_func_cross_entropy)

        loss += bert_loss
        total_loss += loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        
        epochs.set_description("Epoch (Loss=%g)" % round(loss.item(), 5))

    writer.add_scalar('train_loss', total_loss / len(train_data), epoch)
    test_model(model=multi_class_model, tokenizer=tokenizer, epoch=epoch,writer=writer,loss_func=loss_func_cross_entropy)

    # evaluation

    
