import torch
from torch import nn
import torch.nn.functional as F
import logddd
"""
    下游任务的模型，暂时没用
"""
import numpy as np


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.

    This should only be used at test time.

    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.

    Returns:
        viterbi: A [seq_len] list of integers containing the highest scoring tag
                indices.
        viterbi_score: A float containing the score for the Viterbi sequence.
    """
    # 用于存储累计分数的数组
    trellis = np.zeros_like(score)
    # 用于存储最优路径索引的数组
    backpointers = np.zeros_like(score, dtype=np.int32)
    # 第一个时刻的累计分数
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        # 各个状态截止到上个时刻的累计分数 + 转移分数
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        # max（各个状态截止到上个时刻的累计分数 + 转移分数）+ 选择当前状态的分数
        trellis[t] = score[t] + np.max(v, 0)
        # 记录累计分数最大的索引
        backpointers[t] = np.argmax(v, 0)

    # 最优路径的结果
    viterbi = [np.argmax(trellis[-1])]
    # 反向遍历每个时刻，得到最优路径
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


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
        # 标签的类别数量
        self.class_nums = class_nums
        # 全连接网络
        self.fc = nn.Linear(hidden_size, class_nums)
    
    def forward(self, datas):
        labels = datas["labels"]
        # 输入到预训练模型当中去
        outputs = self.bert(**datas)

        # 16 X 128 X 18
        # 取出bert最后一维的hidden_state
        hidden_state = outputs.hidden_states[len(outputs.hidden_states)-1]
        out_fc = self.fc(hidden_state)
        predict_labels = []
        # 遍历每一个句子
        for label_index,sentence_label in enumerate(labels):
            # 遍历句子中的每一个词
            for word_index,val in enumerate(sentence_label):
                # 在当前的label里面，如果值是-100 表示当前位置没有被mask，因此就不需要记录他们的label
                if val != -100:
                    predict_labels.append(out_fc[label_index][word_index].tolist())

        # 这个就是计算出来的label 计算出来的维度是16X18
        predict_labels = torch.tensor(predict_labels)
        # 后面 需要计算出每一个维度的softmax 对应每一个label的概率
        predict_label_ratios = F.softmax(predict_labels,dim=1)
        # 接入维特比算法计算整条链条中整体概率值最大的
        # print(predict_label_ratios)
        score = predict_label_ratios
        transition_params = np.zeros((self.class_nums,self.class_nums))
        viterbi, viterbi_score = viterbi_decode(score,transition_params)
        logddd.log(viterbi,viterbi_score)
        # logddd.log(viterbi_score)
        # logddd.log(predict_label_ratios.shape)
        # 计算过程的公式
        return out_fc
