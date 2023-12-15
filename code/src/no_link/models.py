import torch
import torch.nn.functional as F
from torch import nn
import logddd
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
        return self.nolink(datas)

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
            loss.backward()
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

    def nolink(self, prompts):
        prompt = prompts[0]
        for index, item in enumerate(prompts):
            if index == 0:
                continue
            for k, _ in prompt.items():
                prompt[k] = torch.cat((prompt[k], item[k]), dim=0)

        observe, loss = self.get_score(prompt)
        observe = np.array(observe)
        # observe = np.expand_dims(observe, axis=1)
        total_predict_labels = np.argmax(observe, axis=1)
        # logddd.log(total_predict_labels.shape)
        return total_predict_labels, observe, loss
