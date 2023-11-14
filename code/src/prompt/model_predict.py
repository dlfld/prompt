import sys

import joblib
import logddd
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
# from model_fast import SequenceLabeling
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, BertConfig

from model_params import Config
from models import SequenceLabeling

sys.path.append("..")
from data_process.utils import batchify_list, calcu_loss
from predict import test_model
from data_process.data_processing import load_instance_data
from utils import EarlyStopping

def postag(item):
    model_test, tokenizer_test = load_model(model_checkpoint) 
    test_data_instances,train_data_instances = load_data(item,tokenizer_test)
    test_data = batchify_list(test_data_instances, batch_size=Config.batch_size)
    train_data = batchify_list(train_data_instances, batch_size=Config.batch_size)
    return test_model(item,test_data_instances,train_data_instances,model_test)



def pos_train(item):
    train(item, 0, 0)