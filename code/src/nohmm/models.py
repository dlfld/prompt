import torch
from torch import nn
import torch.nn.functional as F
import logddd
from transformers import AutoTokenizer
from torchcrf import CRF

from model_params import Config
import copy
"""
    下游任务的模型
"""
import numpy as np
import time


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
        # PLB占位符,根据占位符，计算出占位符对应的id
        self.PLB = tokenizer.convert_tokens_to_ids("[PLB]")
        self.total_times = 0

    def forward(self, datas):
        # 取出一条数据,也就是一组prompt,将这一组prompt进行维特比计算
        # 所有predict的label
        total_predict_labels = []
        # 所有的score
        total_scores = []
        # 每一条数据中bert的loss求和
        total_loss = 0
        # 遍历每一个句子生成的prompts
        for data in datas:
            # start_time = time.time()
            # logddd.log(len(data["input_ids"]))
            # scores, seq_predict_labels, loss = self.viterbi_decode(data)
            scores, seq_predict_labels, loss = self.viterbi_decode_v2(data)
            total_predict_labels.append(seq_predict_labels)
            total_scores.append(scores)
            total_loss += loss
            # end_time = time.time()
            # logddd.log(f"总耗时:{end_time - start_time}")
            # logddd.log(f'当前实例生成的prompt数为:{len(data["input_ids"])},运行时间为:{end_time - start_time}')
            # logddd.log(self.total_times)
            # exit(0)
            # del input_data
        return total_predict_labels, total_scores, total_loss / len(datas)
        # return total_predict_labels, total_scores, total_loss

    def get_score(self, prompt):
        """
            将prompt句子放入模型中进行计算，并输出当前的prompt的label矩阵
            @param prompt: 一个prompt句子
            @return score shape-> 1 X class_nums
        """
        # 将每一个数据转换为tensor -> to device
        start_time = time.time()
        prompt = {
            k: torch.tensor(v).to(Config.device)
            for k, v in prompt.items()
        }
        # logddd.log("get_score")
        # 输入bert预训练
        outputs = self.bert(**prompt)

        out_fc = outputs.logits
        loss = outputs.loss
        if loss.requires_grad:
            loss.backward()
        end_time = time.time()
        self.total_times += end_time - start_time
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

        del prompt, outputs, out_fc
        return predict_score, loss.item()

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
        # total_loss_item = 0
        total_loss = 0
        for index in range(seq_nums):
            # 计算出一个prompt的score,求出来的是一个含有一条数据的二维数组，因此需要取[0]
            cur_data = {
                k: [v[index].tolist()]
                for k, v in prompts.items()
            }
            score, loss = self.get_score(cur_data)
            # start_time = time.time()
            if loss is not None:
                total_loss += loss
                # 每8次计算一下梯度
                # if index % 7 == 0 and loss.requires_grad:
                #     total_loss.backward()
                # del loss,total_loss
                # total_loss = 0
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
                        item = trellis[index - 1][trellis_idx] + self.transition_params[trellis_idx][score_idx] + \
                               score[score_idx]
                        temp.append(item.item())
                    # trellis[index-1] + self.transition_params[trellis_idx] + score
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
                next_prompt = torch.tensor([x if x != self.PLB else cur_predict_label_id for x in next_prompt])
                # logddd.log(next_prompt == prompts[index + 1])
                prompts["input_ids"][index + 1] = next_prompt
            # end_time = time.time()
            # self.times += end_time - start_time
        # pre_index 记录的是每一步的路径来源，取出最后一列最大值对应的来源路径
        seq_predict_labels = pre_index[-1][np.argmax(trellis[-1])]

        # return trellis, seq_predict_labels, total_loss / seq_nums
        # logddd.log(F.softmax(torch.tensor(trellis)).shape)
        # logddd.log(seq_predict_labels)
        return F.softmax(torch.tensor(trellis)),seq_predict_labels,total_loss / seq_nums
        # return trellis,seq_predict_labels,total_loss / seq_nums


    def viterbi_decode_v2(self, prompts):
        total_loss = 0
        seq_len, num_labels = len(prompts["input_ids"]), len(self.transition_params)
        labels = np.arange(num_labels).reshape((1, -1))
        scores = None
        paths = labels
        # logddd.log(seq_len)
        trellis = None
        for index in range(seq_len):
            cur_data = {
                k: [v[index].tolist()]
                for k, v in prompts.items()
            }

            observe, loss = self.get_score(cur_data)
            observe = np.array(observe[0])
            # start_time = time.time()
            # 当前轮对应值最大的label
            cur_predict_label_id = None
            # loss 叠加
            total_loss += loss
            if index == 0:
                # 第一个句子不用和其他的进行比较，直接赋值
                trellis = observe.reshape((1, -1))
                scores = observe
                cur_predict_label_id = np.argmax(observe)
            else:
                M = scores + self.transition_params.cpu().detach().numpy() + observe
                scores = np.max(M, axis=0).reshape((-1, 1))
                # shape一下，转为列，方便拼接和找出最大的id(作为预测的标签)
                shape_score = scores.reshape((1,-1))
                # 添加过程矩阵，后面求loss要用
                trellis = np.concatenate([trellis,shape_score],0)
                # 计算出当前过程的label
                cur_predict_label_id = np.argmax(shape_score)
                idxs = np.argmax(M, axis=0)
                paths = np.concatenate([paths[:, idxs], labels], 0)
            # 如果当前轮次不是最后一轮，那么我们就
            if index != seq_len - 1:
                next_prompt = prompts["input_ids"][index + 1]
                next_prompt = torch.tensor([x if x != self.PLB else cur_predict_label_id for x in next_prompt])
                # logddd.log(next_prompt == prompts[index + 1])
                prompts["input_ids"][index + 1] = next_prompt


        best_path = paths[:, scores.argmax()]
        # 这儿返回去的是所有的每一句话的平均loss
        return F.softmax(torch.tensor(trellis)),best_path,total_loss / seq_len