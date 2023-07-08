import torch
from torch import nn
import torch.nn.functional as F
import logddd

"""
    下游任务的模型，暂时没用
"""
import numpy as np


class SequenceLabeling(nn.Module):
    def __init__(self, bert_model, hidden_size, class_nums, tokenizer):
        """
            @param bert_model: 预训练模型
            @param hidden_size: 隐藏层大小
            @param class_nums: 类别数
            @param tokenizer:tokenizer
        """
        super(SequenceLabeling, self).__init__()
        # bert 模型
        self.bert = bert_model
        # 标签的类别数量
        self.class_nums = class_nums
        # 全连接网络
        self.fc = nn.Linear(hidden_size, class_nums)
        # 定义维特比算法的trans数组，这个数组是可学习的参数
        self.transition_params = nn.Parameter(torch.randn(class_nums, class_nums))
        # tokenizer
        self.tokenizer = tokenizer

    def forward(self, datas):
        # 输入到预训练模型当中去
        outputs = self.bert(**datas)
        # 16 X 128 X 18
        # 取出bert最后一维的hidden_state
        hidden_state = outputs.hidden_states[len(outputs.hidden_states) - 1]
        out_fc = self.fc(hidden_state)

        predict_labels = []
        # 遍历每一个句子 抽取出被mask位置的隐藏向量,后面送入softmax中
        for label_index, sentences in enumerate(datas["input_ids"]):
            # 遍历句子中的每一词,
            for word_index, val in enumerate(sentences):
                if val == self.tokenizer.mask_token_id:
                    predict_labels.append(out_fc[label_index][word_index].tolist())

        # ==============================viterbi=============================
        # 这个就是计算出来的label 计算出来的维度是16X18
        predict_labels = torch.tensor(predict_labels)
        # 后面 需要计算出每一个维度的softmax 对应每一个label的概率
        # 交叉熵内置了softmax
        # score = F.softmax(predict_labels, dim=1)
        score = predict_labels
        # 计算梯度
        score.requires_grad = True
        # 获取维特比的输出结果，将路径中的每一个概率获取到
        viterbi, viterbi_score = self.viterbi_decode(score)
        viterbi_score = torch.tensor(viterbi_score, requires_grad=True)
        return score, viterbi, viterbi_score

    # ==============================viterbi=============================

    def viterbi_decode(self, score):
        """
         维特比算法，计算当前结果集中的最优路径
        Args:
            score: A [seq_len, num_tags] matrix of unary potentials.

        Returns:
            viterbi: A [seq_len] list of integers containing the highest scoring tag
                    indices.
            viterbi_score: A float containing the score for the Viterbi sequence.
        """
        # 用于存储累计分数的数组
        cur_score = score.detach().numpy()
        trellis = np.zeros_like(cur_score)
        # 用于存储最优路径索引的数组
        backpointers = np.zeros_like(cur_score, dtype=np.int32)
        # 第一个时刻的累计分数
        trellis[0] = cur_score[0]

        for t in range(1, cur_score.shape[0]):
            # 各个状态截止到上个时刻的累计分数 + 转移分数
            # 这一步骤是将[xx,xx,xx,..] 18维的向量变成[[xx],[xx],...]这样的形状
            v = np.expand_dims(trellis[t - 1], 1) + self.transition_params.detach().numpy()
            # 转移矩阵，科学系参数
            # v += self.transition_params
            # max（各个状态截止到上个时刻的累计分数 + 转移分数）+ 选择当前状态的分数
            trellis[t] = cur_score[t] + np.max(v, 0)
            # 记录累计分数最大的索引
            backpointers[t] = np.argmax(v, 0)

        # 最优路径的结果 找到总的参数值最大的一个
        viterbi = [np.argmax(trellis[-1])]
        # 反向遍历每个时刻，得到最优路径
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = np.max(trellis[-1])
        return viterbi, viterbi_score
