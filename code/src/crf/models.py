import sys

import torch
import torch.nn.functional as F
from torch import nn
from torchcrf import CRF
import logddd

sys.path.append("..")
from model_params import Config


class CRFModel(nn.Module):
    def __init__(self, bert_model, class_nums, tokenizer, bert_config, model_checkpoint):
        """
            @param bert_model: 预训练模型
            @param hidden_size: 隐藏层大小
            @param class_nums: 类别数
            @param tokenizer:tokenizer
        """
        super(CRFModel, self).__init__()
        # bert 模型
        self.bert = bert_model
        # 标签的类别数量
        self.class_nums = class_nums

        self.embed_size = 0
        if "bert_" in model_checkpoint or "bart" in model_checkpoint:
            self.embed_size = 1024
        else:
            self.embed_size = 768

        self.model_checkpoint = model_checkpoint

        self.fc = nn.Linear(self.embed_size, Config.class_nums)

        self.tokenizer = tokenizer

        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.crf = CRF(self.class_nums, batch_first=True)

    def forward(self, datas):
        # logddd.log(datas)
        inputs = {
            "input_ids": datas["input_ids"],
            "attention_mask": datas["attention_mask"],
        }
        if "bart" in self.model_checkpoint:
            output = self.bert(**inputs).last_hidden_state
        else:
            output = self.bert(**inputs)[0]

        # 5 128 18
        seq_out = self.fc(output)

        labels = datas["labels"] if "labels" in datas.keys() else None

        path = self.crf.decode(seq_out, mask=datas["attention_mask"].bool())
        res_paths = [x + 1 for row in path for x in row]
        loss = 0
        if labels is not None:
            label = []
            for sentence in datas["labels"]:
                # item = [x - 1 if x != -100 else 0 for x in sentence.tolist()]
                # 模型加载的label是比原始的大1
                item = [x - 1 if x != -100 else 0 for x in sentence.tolist()]
                label.append(item)

            loss = -self.crf(seq_out, torch.tensor(label).to(Config.device), mask=datas["attention_mask"].byte(),
                             reduction='mean')
        return 0, loss, res_paths

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

        predictions = torch.argmax(probabilities, dim=1)
        predictions = predictions.tolist()
        predictions = [x + 1 for x in predictions]
        return predictions

