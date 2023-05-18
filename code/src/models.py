import torch
from torch import nn
import logddd
class MultiClass(nn.Module):
    def __init__(self,bert_model,hidden_size,class_nums):
        """
            @param bert_model: 预训练模型
            @param hidden_size: 隐藏层大小
            @param class_nums: 类别数
        """
        super(MultiClass, self).__init__()
        # bert 模型
        self.bert = bert_model
        # 全连接网络
        self.fc = nn.Linear(hidden_size, class_nums)
    
    def forward(self, batch):
        outputs = self.bert(**batch)
        logits = outputs.logits
        logddd.log(logits.shape)
        out_fc = self.fc(logits)
        return out_fc
