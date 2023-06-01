import logddd
import torch
from torch import nn
from torchcrf import CRF
import sys
import torch.nn.functional as F

sys.path.append("..")
from model_params import Config


class CRFModel(nn.Module):
    def __init__(self, bert_model, class_nums, tokenizer):
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
        self.crf = CRF(num_tags=Config.class_nums, batch_first=True)
        # tokenizer
        self.tokenizer = tokenizer

    def forward(self, datas):
        output = self.bert(**datas)
        loss = output.loss
        # batch 128 21128
        logits = output.logits
        # 记录那些位置是填充进取的
        masks = []
        for mask in datas["attention_mask"]:
            masks.append(mask.tolist())
        masks_crf = torch.tensor(masks, dtype=torch.bool).to(Config.device)
        res_logits = logits[:, :, 1:1 + Config.class_nums]
        labels = []
        for sentence in datas["labels"]:
            item = [x if x != -100 else 0 for x in sentence.tolist()]
            labels.append(item)

        datas["labels"] = torch.tensor(labels).to(Config.device)

        crf_loss = self.crf(res_logits, datas["labels"], mask=masks_crf, reduction="mean")
        return loss - crf_loss
        # return loss, crf_loss
