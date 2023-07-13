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
        # PLB占位符,根据占位符，计算出占位符对应的id
        self.PLB = tokenizer.convert_tokens_to_ids("[PLB]")

    def forward(self, datas):
        # # 取出一条数据,也就是一组prompt,将这一组prompt进行维特比计算
        # # 所有predict的label
        # total_predict_labels = []
        # # 所有的score
        # total_scores = []
        # # 每一条数据中bert的loss求和
        # total_loss = 0

        # total_data = []
        # for data in datas:
        #     # input_data = {
        #     #     # k: v.to(Config.device)
        #     #     k: v[:8]
        #     #     for k, v in data.items()
        #     # }
        #     scores, seq_predict_labels, loss = self.viterbi_decode(data)
        #     total_predict_labels.append(seq_predict_labels)
        #     total_scores.append(scores)
        #     total_loss += loss
        #     # del input_data
        total_scores,total_predict_labels,total_loss = self.viterbi_decode(datas)

        return total_predict_labels, total_scores, total_loss 
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
        # logddd.log("get_score")
        # 输入bert预训练
        outputs = self.bert(**prompt)
        out_fc = outputs.logits
        loss = outputs.loss
        if loss.requires_grad:
            # logddd.log("backward")
            loss.backward()
        # logddd.log(loss)
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

    def viterbi_decode(self, batch_prompts):

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
        max_len = 0
        # 存当前batch的预测路径
        total_predict_labels = [[0]] * len(batch_prompts)
        total_labels = []
        total_paths = []
        # 存当前batch的score
        total_scores = []
        # 当前batch中所有句子的trellis数组
        total_trellis = []
        total_cur_predict_label_id = []
        # 记录总的prompt有多少条
        prompt_nums = 0
        # 遍历每一个组prompt句子，也就是遍历batch中的每一个原句
        for seq_prompts in batch_prompts:
            seq_len = len(seq_prompts["input_ids"])
            max_len = max(seq_len, max_len)
            prompt_nums += seq_len
    
        total_loss = 0
        # 遍历batch内最长的句子j，将batch内每一个句子中的prompt都组织起来，
        for index in range(max_len):
            # index 代表的是prompt内的第几个句子
            step = None

            # 遍历一个batch那的每一个句子,将对应位置上的进行合并
            for batch_idx, seq_prompts in enumerate(batch_prompts):
                # logddd.log(seq_prompts)
                # exit(0)
                # 如果是第一个，那就初始化step
                if batch_idx == 0:
                    step = {
                        # 如果当前句子长度小于当前句子的最长长度，那就是用当前setp的prompt，如果当前长度大于当前句子的最大长度，那就用句子的最后一个prompt占位
                        k: [v[index].tolist()] if index < len(v) else [v[len(v) - 1].tolist()]
                        for k, v in seq_prompts.items()
                    }
                else:
                    # 如果不是第一个就找到指定位置的prompt加进去
                    for k, v in seq_prompts.items():
                        step[k].append(v[index].tolist() if index < len(v) else v[len(v) - 1].tolist())
            # logddd.log("进来了")
            scores, loss = self.get_score(step)

            if loss is not None:
                total_loss += loss
                del loss
            
            # 遍历计算出来的score
            for score_idx,observe in enumerate(scores):
                observe = np.array(observe)
                
                # scores 是当前这一个step上所有prompt的计算结果
                # score_idx 句子对应的编号，（一个batch中句子对应的编号）
                if index >= len(batch_prompts[score_idx]["input_ids"]):
                    # 如果当前step已经超过了当前句子的长度，那就不需要算Viterbi了，直接跳过
                    continue
                # 如果当前step是第一轮
                if index == 0:
                    num_labels = len(self.transition_params)
                    labels = np.arange(num_labels).reshape((1, -1))

                    total_labels.append(labels)
                    total_paths.append(labels)
                    total_trellis.append(observe.reshape((1, -1)))
                    total_scores.append(observe)
                    total_cur_predict_label_id.append(np.argmax(observe))
                else:
                    M = total_scores[score_idx] + self.transition_params.cpu().detach().numpy() + observe
                    # 取出当前维特比的scores
                    scores_item = total_scores[score_idx]
                    scores_item = np.max(M, axis=0).reshape((-1, 1))
                    total_scores[score_idx] = scores_item

                    shape_score = scores_item.reshape((1,-1))
                    total_trellis[score_idx] =  np.concatenate([total_trellis[score_idx],shape_score],0)
                    total_cur_predict_label_id[score_idx] = np.argmax(shape_score)
                    idxs = np.argmax(M, axis=0)
                    total_paths[score_idx] = np.concatenate([total_paths[score_idx][:, idxs], labels], 0)
                
                # 如果当前轮次不是当前句子的最后一轮
                if index != len(batch_prompts[score_idx]["input_ids"])-1:
                    next_prompt = batch_prompts[score_idx]["input_ids"][index + 1]
                    next_prompt = torch.tensor([x if x != self.PLB else total_cur_predict_label_id[score_idx] for x in next_prompt])
                    batch_prompts[score_idx]["input_ids"][index + 1] = next_prompt
                
                # 如果当前轮次是其中某些句子的最后一轮，那么就要计算最好的path，并将score矩阵加入到最终的score矩阵中去
                if index == len(batch_prompts[score_idx]["input_ids"])-1:
                    total_predict_labels[score_idx] = total_paths[score_idx][:,total_scores[score_idx].argmax()]
                    total_trellis[score_idx] = F.softmax(torch.tensor(total_trellis[score_idx]))


        return total_trellis,total_predict_labels,total_loss/prompt_nums
    