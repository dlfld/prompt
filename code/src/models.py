import torch
from torch import nn
import torch.nn.functional as F
import logddd
"""
    下游任务的模型，暂时没用
"""
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
        logddd.log(predict_label_ratios.shape)
        for item in predict_label_ratios:
            print(sum(item.tolist()))
        # 计算过程的公式
        return out_fc
