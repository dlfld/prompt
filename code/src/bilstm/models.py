import logddd
import torch
from torch import nn
from torchcrf import CRF
import sys
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

sys.path.append("..")
from model_params import Config


class BiLSTMCRFModel(nn.Module):
    def __init__(self, bert_model, class_nums, tokenizer, bert_config):
        """
            @param bert_model: 预训练模型
            @param hidden_size: 隐藏层大小
            @param class_nums: 类别数
            @param tokenizer:tokenizer
        """
        super(BiLSTMCRFModel, self).__init__()
        # bert 模型
        self.bert = bert_model
        # 标签的类别数量
        self.class_nums = class_nums

        # bilstm
        self.dropout = 0.2
        self.dropout1 = nn.Dropout(p=self.dropout)
        # self.lstm = nn.LSTM(Config.class_nums, 21129, num_layers=2, bidirectional=True, batch_first=True)
        # tokenizer
        self.hidden_size = 128
        self.lstm = nn.LSTM(input_size=1024, hidden_size=self.hidden_size, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=self.dropout)
        # self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        # fc
        self.fc = nn.Linear(self.hidden_size * 2, Config.class_nums)

        self.tokenizer = tokenizer
        self.rnn_layers = 1
        # self.cls = BertOnlyMLMHead(bert_config)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, datas):
        # logddd.log(datas)
        inputs = {
            "input_ids": datas["input_ids"],
            "attention_mask": datas["attention_mask"],
        }
        output = self.bert(**inputs)[0]
        seq_out, _ = self.lstm(output)

        # seq_out = seq_out.contiguous().view(-1, self.hidden_size * 2)
        # seq_out = seq_out.contiguous().view(Config.batch_size, Config.sentence_max_len, -1)
        logits = self.fc(seq_out)
        labels = datas["labels"]
        loss = self.get_loss(logits, labels)
        return logits, loss, self.get_paths(logits, labels)

    def get_loss(self, logits, labels):
        outputs = logits.view(-1, self.class_nums)
        label = labels.view(-1)
        for i in range(len(label)):
            if label[i] != -100:
                label[i] = label[i] - 1
        masked_lm_loss = self.loss_fct(outputs, labels.view(-1))
        return masked_lm_loss

    def get_paths(self, logits, labels):
        # logddd.log(logits.shape)
        total_labels = []
        for s_idx, sentence in enumerate(labels):
            for w_idx, item in enumerate(sentence):
                if item != -100:
                    total_labels.append(logits[s_idx][w_idx].tolist())

        probabilities = F.softmax(torch.tensor(total_labels).to(Config.device))
        # logddd.log(probabilities.shape)
        predictions = torch.argmax(probabilities, dim=1)
        predictions = predictions.tolist()
        predictions = [x + 1 for x in predictions]
        return predictions
