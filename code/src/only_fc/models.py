import torch
from torch import nn
from transformers import AutoTokenizer

import torch.nn.functional as F
from model_params import Config
import logddd

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
        # self.transition_params = nn.Parameter(torch.randn(class_nums, class_nums, requires_grad=True).to(Config.device) )
        # tokenizer
        self.tokenizer = tokenizer

    def forward(self, datas):
        # for prompts in datas:
        #     for i in range(len(prompts)):
        #         prompts[i] = {
        #             k: v.tolist() if type(v) == torch.Tensor else v
        #             for k,v in prompts[i].items()
        #         }
        #     cur_data = prompts[0]
        #     for i in range(1,len(prompts)):
        #         cur_data["input_ids"] = cur_data["input_ids"]+ prompts[i]["input_ids"]
        #         # cur_data["attention_mask"] = torch.cat(cur_data["attention_mask"],prompts[i]["attention_mask"],dim=0)
        #         cur_data["attention_mask"] = cur_data["attention_mask"] + prompts[i]["attention_mask"]
        #         # cur_data["attention_mask"].append(prompts[i]["attention_mask"])
        #         if "labels" in cur_data.keys():
        #             cur_data["labels"] = cur_data["labels"] + prompts[i]["labels"]
        #     cur_data = {
        #         k:torch.tensor(v)
        #         for k,v in cur_data.items()
        #     }
        #     out_fc, loss = self.get_score(cur_data)
        #
        #     # out_fc = out_fc.tolist()
        #     path = []
        #
        #     for item in out_fc:
        #         path.append(np.argmax(item))
        # total_loss = torch.tensor(0,dtype=torch.float).to(Config.device)
        total_loss = 0
        paths = []
        out_fcs = []
        batch_fc = []
        batch_path = []
        for prompts in datas:
            for index,prompt in enumerate(prompts):
                out_fc,loss = self.get_score(prompt)
                # out_fc = out_fc[0]
                if loss != None:
                    total_loss += loss
                    logddd.log(total_loss)
                cur_label = np.argmax(out_fc[1:19])

                paths.append(cur_label)
                out_fcs.append(out_fc[1:19])

                # 如果当前的句子不是最后一条,那就将当前句子的结果填充到下一条句子中
                if index != len(prompts) - 1:
                    next_prompt = prompts[index + 1]["input_ids"]
                    # 21指的是，上一个句子预测出来的词性的占位值，将占位值替换成当前句子预测出来的值
                    next_prompt[next_prompt == 21] = cur_label
                    prompts[index + 1]["input_ids"] = next_prompt

            batch_fc.append(out_fcs)
            batch_path.append(paths)
        # logddd.log(total_loss)
        return batch_path,batch_fc,total_loss




    def get_score(self, prompt):

        """
            将prompt句子放入模型中进行计算，并输出当前的prompt的label矩阵
            @param prompt: 一个prompt句子
            @return score shape-> 1 X class_nums
        """
        # model_checkpoint = Config.model_checkpoint
        # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        # tokenizer.add_special_tokens({'additional_special_tokens': ["[PLB]"]})
        # print("输入","".join(tokenizer.convert_ids_to_tokens(crf["input_ids"][0])))
        prompt = {
            k: v.to(Config.device)
            for k, v in prompt.items()
        }
        outputs = self.bert(**prompt)
        out_fc = outputs.logits
        loss = outputs.loss

        if loss != None:
            loss = loss.tolist()
            logddd.log(loss)
        # print(out_fc.shape)

        # return out_fc
        # out_fc = outputs
        # 获取到mask维度的label
        predict_labels = []
        # # 遍历每一个句子 抽取出被mask位置的隐藏向量,后面送入softmax中
        for label_index, sentences in enumerate(prompt["input_ids"]):
            # 遍历句子中的每一词,
            for word_index, val in enumerate(sentences):
                if val == self.tokenizer.mask_token_id:
                    predict_labels.append(out_fc[label_index][word_index].tolist())
        # exit(0)
        predict_labels = predict_labels[0]


        return predict_labels,loss

    def viterbi_decode(self, prompts):
        """
         维特比算法，计算当前结果集中的最优路径
        @param prompts: 一组prompt句子
        @return
            loss_value: 维特比每一步的最大值的求和
            seq_predict_labels:记录每一步骤预测出来的标签的值
            trellis: 存储累计得分的数组
        """
        # 进入维特比算法，挨个计算
        # 存储累计得分的数组
        trellis = np.zeros((len(prompts), self.class_nums))
        pre_index = []

        # 损失
        loss_value = 0
        for index in range(len(prompts)):
            prompt = prompts[index]
            prompt = {
                k: v
                for k, v in prompt.items()
            }
            # 计算出一个prompt的score
            score = self.get_score(prompt)[0]
            # 如果是第一个prompt句子
            if index == 0:
                # 第一个句子不用和其他的进行比较，直接赋值
                trellis[0] = score.detach().numpy()
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
                    for trellis_idx in range(len(trellis[index - 1])):
                        item = trellis[index - 1][trellis_idx] + self.transition_params[trellis_idx][score_idx] + score[
                            score_idx]
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

            # ======================================================
            # 最优路径的结果 找到总的参数值最大的一个，index。加一是因为index和词表的index做对应
            # 下标的最大值
            cur_predict_label_id = np.argmax(trellis[index]) + 1
            # 当前步骤中，找到最大值，这个value要作为计算后面的loss
            loss_value += np.max(trellis[index])
            # 如果当前的句子不是最后一条,那就将当前句子的结果填充到下一条句子中
            if index != len(prompts) - 1:
                next_prompt = prompts[index + 1]["input_ids"]
                # 21指的是，上一个句子预测出来的词性的占位值，将占位值替换成当前句子预测出来的值
                next_prompt[next_prompt == 21] = cur_predict_label_id
                # logddd.log(next_prompt == prompts[index + 1])
                prompts[index + 1]["input_ids"] = next_prompt

        # pre_index 记录的是每一步的路径来源，取出最后一列最大值对应的来源路径
        seq_predict_labels = pre_index[-1][np.argmax(trellis[-1])]

        return loss_value, seq_predict_labels, trellis
