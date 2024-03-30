import torch
from torch import nn

from model_params import Config

"""
    下游任务的模型
"""

from torchcrf import CRF

class SequenceLabeling(nn.Module):

    def __init__(self, bert_model, hidden_size, class_nums, tokenizer, model_checkpoint):
        """
            @param bert_model: 预训练模型
            @param hidden_size: 隐藏层大小
            @param class_nums: 类别数
            @param tokenizer:tokenizer
        """
        super(SequenceLabeling, self).__init__()
        # bert 模型
        self.bert = bert_model.to(Config.device)
        # 标签的类别数量
        self.class_nums = class_nums
        # 全连接网络
        self.fc = nn.Linear(hidden_size, class_nums)
        # 定义维特比算法的trans数组，这个数组是可学习的参数
        self.transition_params = nn.Parameter(torch.randn(class_nums, class_nums, requires_grad=True).to(Config.device))
        # tokenizer
        self.tokenizer = tokenizer
        # PLB占位符,根据占位符，计算出占位符对应的id
        self.PLB = tokenizer.convert_tokens_to_ids("[PLB]")
        self.total_times = 0
        # 当前所有标签的embedding
        """
            @param bert_model: 预训练模型
            @param hidden_size: 隐藏层大小
            @param class_nums: 类别数
            @param tokenizer:tokenizer
        """
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
        self.embed_size = 0
        if "bert_" in model_checkpoint or "bart" in model_checkpoint:
            self.embed_size = 1024
        else:
            self.embed_size = 768

        self.model_checkpoint = model_checkpoint
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=self.dropout)
        # fc
        self.fc = nn.Linear(self.hidden_size * 2, Config.class_nums)

        self.tokenizer = tokenizer
        # self.rnn_layers = 1
        # self.cls = BertOnlyMLMHead(bert_config)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.crf = CRF(self.class_nums, batch_first=True)
        # self.labels_embeddings = self.get_label_embeddings()

    #
    def forward(self, datas):
        # 取出一条数据,也就是一组prompt,将这一组prompt进行维特比计算
        # 所有predict的label
        total_predict_labels = []
        # 所有的score
        total_scores = []
        # 每一条数据中bert的loss求和
        total_loss = 0
        # 遍历每一个句子生成的prompts
        # logddd.log(self.transition_params.tolist())
        # logddd.log(len(datas))
        for index, data in enumerate(datas):
            # self.viterbi_decode_v2(data)
            scores, seq_predict_labels, loss = self.get_score(data)
            total_predict_labels.append(seq_predict_labels)
            total_scores.append(scores)
            total_loss += loss

        return total_predict_labels, total_scores, total_loss / len(datas)
        # return total_predict_labels, total_scores, total_loss

    def get_score(self, prompt):
        """
            将prompt句子放入模型中进行计算，并输出当前的prompt的label矩阵
            @param prompt: 一个prompt句子
            @return score shape-> 1 X class_nums
        """
        # 将每一个数据转换为tensor -> to device
        prompt = {
            k: torch.tensor(v).to(Config.device)
            for k, v in prompt.items()
        }
        # 输入bert预训练
        if "bart" in self.model_checkpoint:
            output = self.bert(**prompt).last_hidden_state
        else:
            output = self.bert(**prompt)[0]
        seq_out, _ = self.lstm(output)
        seq_out = self.fc(seq_out)
        labels = prompt["labels"]

        path = self.crf.decode(seq_out, mask=prompt["attention_mask"].bool())
        res_paths = [x + 1 for row in path for x in row]
        loss = 0
        if labels is not None:
            label = []
            for sentence in prompt["labels"]:
                # 模型加载的label是比原始的大1
                item = [x - 1 if x != -100 else 0 for x in sentence.tolist()]
                label.append(item)
            loss = -self.crf(seq_out, torch.tensor(label).to(Config.device), mask=prompt["attention_mask"].byte(),
                             reduction='sum')
        return res_paths, 0, loss / Config.batch_size,
