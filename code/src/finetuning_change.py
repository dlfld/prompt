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
from data_process.data_processing import load_data
class Config(object):
    """
        配置类，保存配置文件
    """
    # 训练集位置
    train_dataset_path = "/home/dlf/prompt/code/data/jw/short_data_train.txt"
    # 测试集位置
    test_dataset_path = ""
    # prompt dataset
    train_dataset_path = "/home/dlf/prompt/dataset.csv"
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

# 加载训练数据
standard_data = load_data(Config.train_dataset_path)
