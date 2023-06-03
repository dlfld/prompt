import logddd
import torch
from torch import nn
from torchcrf import CRF
import sys
import torch.nn.functional as F

sys.path.append("..")
from model_params import Config


class BiLSTMCRFModel(nn.Module):
    def __init__(self, bert_model, class_nums, tokenizer):
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
        # crf
        self.crf = CRF(num_tags=Config.class_nums, batch_first=True)
        # bilstm
        self.dropout = nn.Dropout(0.5)
        rnn_dim = 128
        out_dim = Config.class_nums * 2
        self.bilstm = nn.LSTM(Config.class_nums,rnn_dim,num_layers=1,bidirectional=True,batch_first=True)
        # tokenizer
        self.tokenizer = tokenizer
        self.hidden2tag = nn.Linear(out_dim,Config.class_nums)

    def forward(self, datas):
        # logddd.log(datas)
        output = self.bert(**datas)
        loss = output.loss
        # batch 128 21128
        logits = output.logits
        # 记录那些位置是填充进取的
        masks = []
        for mask in datas["attention_mask"]:
            masks.append(mask.tolist())
        # 转换为tensor
        masks_crf = torch.tensor(masks, dtype=torch.bool).to(Config.device)
        # bert的输出是21128维的，截取词性标签在词表中index的那一段，拿出来用,这个就当作是emissions矩阵
        res_logits = logits[:, :, 1:1 + Config.class_nums]
        # 到此位置就拿到了bert的输出
        lstm_output,_ = self.bilstm(res_logits)
        lstm_output = self.dropout(lstm_output)
        emissions = self.hidden2tag(lstm_output)
        logddd.log(emissions.shape)
        exit(0)
        # 将label中填充的-100转换成0，因为crf中只有设置的label数量，放-100进取会报错
        labels = []
        for sentence in datas["labels"]:
            item = [x - 1 if x != -100 else 0 for x in sentence.tolist()]
            labels.append(item)
        # 写回label
        datas["labels"] = torch.tensor(labels).to(Config.device)
        crf_loss = self.crf(res_logits, datas["labels"], mask=masks_crf, reduction="mean")
        # 将bert的loss和crf的loss加起来，因为crfloss是负对数似然函数，因此在这个地方取负
        total_loss = loss - crf_loss
        # 获取crf计算出来的最优路径
        decode = self.crf.decode(res_logits, mask=masks_crf)
        # 预测出来的label,和真实label之间相差1，因为在词表当中，真实label的id是从1开始，因此需要加1
        predict_labels = []
        for sequence in decode:
            predict_labels.append([x + 1 for x in sequence])

        return total_loss, predict_labels
