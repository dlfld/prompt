import torch
from torch import nn
import torch.nn.functional as F
import logddd
from transformers import AutoTokenizer
from torchcrf import CRF

from model_params import Config

"""
    下游任务的模型
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
        self.transition_params = nn.Parameter(torch.randn(class_nums, class_nums, requires_grad=True).to(Config.device))
        # tokenizer
        self.tokenizer = tokenizer

    def forward(self, datas):
        # 取出一条数据,也就是一组prompt,将这一组prompt进行维特比计算
        # 所有predict的label
        total_predict_labels = []
        # 所有的score
        total_scores = []
        # 每一条数据中bert的loss求和
        total_loss = 0
        # print(datas)
        # exit(0)
        # 遍历每一个句子生成的prompts
        for data in datas:
            input_data = {
                # k: v.to(Config.device)
                k: v
                for k, v in data.items()
            }
            scores, seq_predict_labels, loss = self.viterbi_decode(input_data)
            total_predict_labels.append(seq_predict_labels)
            total_scores.append(scores)
            total_loss += loss
            del input_data

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
        outputs = self.bert(**prompt)
        out_fc = outputs.logits
        # 获取到mask维度的label
        predict_labels = []
        # 遍历每一个句子 抽取出被mask位置的隐藏向量, 也就是抽取出mask
        for label_index, sentences in enumerate(prompt["input_ids"]):
            # 遍历句子中的每一词,
            for word_index, val in enumerate(sentences):
                if val == self.tokenizer.mask_token_id:
                    predict_labels.append(out_fc[label_index][word_index].tolist())
        # 获取指定位置的数据
        predict_score = [score[1:1 + Config.class_nums] for score in predict_labels]
        del prompt
        return predict_score, outputs.loss

    def viterbi_decode(self, prompts):
        """
         维特比算法，计算当前结果集中的最优路径
        @param prompts: 一组prompt句子
        @return
            scores: 模型输出的概率矩阵
            seq_predict_labels:记录每一步骤预测出来的标签的值
            loss: 预训练模型给出的loss
        """
        # 进入维特比算法，挨个计算
        # 如果当前是训练模式，那么就不需要经过维特比算法
        # if Config.model_train:
        #     # 预测出来的scores数组,
        #     scores,loss = self.get_score(prompts)
        #     return scores,[],loss
        # 如果当前是测试模式
        # 当前句子的数量
        seq_nums = len(prompts["input_ids"])
        # 存储累计得分的数组

        trellis = np.zeros((seq_nums, self.class_nums))

        # 存放路径的列表
        pre_index = []
        total_loss = 0
        for index in range(seq_nums):
            # 计算出一个prompt的score,求出来的是一个含有一条数据的二维数组，因此需要取[0]
            cur_data = {
                k: [v[index].tolist()]
                for k, v in prompts.items()
            }
            print(len(cur_data["input_ids"]))
            score, loss = self.get_score(cur_data)
            if loss is not None:
                total_loss += loss
                del loss
            # 预测的时候是一条数据一条数据d
            score = score[0]
            # 如果是第一个prompt句子
            if index == 0:
                # 第一个句子不用和其他的进行比较，直接赋值
                trellis[0] = score
                # 如果是第一个节点，那么当前节点的位置来自于自己
                pre_index.append([[i] for i in range(len(trellis[0]))])
            # =======================================================
            else:
                trellis_cur = []
                pre_index.append([[i] for i in range(self.class_nums)])
                for score_idx in range(self.class_nums):
                    # 记录的是前面一个步骤的每一个节点到当前节点的值
                    temp = []
                    # 计算当前步骤中，前一个步骤每一个标签到第score_index个标签的值
                    for trellis_idx in range(self.class_nums):
                        # 这里暂时设置transition为1矩阵
                        # item = trellis[index - 1][trellis_idx] * score[score_idx]
                        item = trellis[index - 1][trellis_idx] + self.transition_params[trellis_idx][score_idx] + \
                               score[score_idx]
                        temp.append(item.item())
                    temp = np.array(temp)
                    # 最大值
                    max_value = np.max(temp)
                    # 最大值下标
                    max_index = np.argmax(temp)
                    # 记录当前节点的前一个节点位置
                    pre_index[index][score_idx] = pre_index[index - 1][max_index] + [score_idx]
                    # logddd.log(pre_index)
                    trellis_cur.append(max_value)
                # 记录当前时刻的值
                trellis[index] = np.array(trellis_cur)

            # 下标的最大值
            cur_predict_label_id = np.argmax(trellis[index]) + 1
            # 将下一句的占位符替换为当前这一句的预测结果，在训练过程中因为训练数据没有添加占位符，因此不会替换
            if index != seq_nums - 1:
                next_prompt = prompts["input_ids"][index + 1]
                # 21指的是，上一个句子预测出来的词性的占位值，将占位值替换成当前句子预测出来的值
                # next_prompt[next_prompt == 21] = cur_predict_label_id
                next_prompt = torch.tensor([x if x != 21 else cur_predict_label_id for x in next_prompt])
                # logddd.log(next_prompt == prompts[index + 1])
                prompts["input_ids"][index + 1] = next_prompt

        # pre_index 记录的是每一步的路径来源，取出最后一列最大值对应的来源路径
        seq_predict_labels = pre_index[-1][np.argmax(trellis[-1])]
        # total_loss / seq_nums 当前的total_loss是bert内部的loss，对每一个prompt求loss，然后求平均
        return trellis, seq_predict_labels, total_loss / seq_nums
