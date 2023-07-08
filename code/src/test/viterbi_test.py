import logddd
import numpy as np


class Viterbi_test:
    def __int__(self):
        self.class_nums = 2

    def viterbi_decode(self, prompts, scores, transition):
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
        class_nums = 2
        trellis = np.zeros((len(prompts), class_nums))
        pre_index = []
        # 记录每一步骤预测出来的标签的值
        seq_predict_labels = []

        # 损失
        loss_value = 0
        for index in range(len(prompts)):
            # 计算出一个prompt的score
            score = scores[index]
            # 如果是第一个prompt句子
            if index == 0:
                # 第一个句子不用和其他的进行比较，直接赋值
                trellis[0] = score
                # 如果是第一个节点，那么当前节点的位置来自于自己
                pre_index.append([[i] for i in range(len(trellis[0]))])
            # =======================================================
            else:
                trellis_cur = []
                pre_index.append([[i] for i in range(class_nums)])
                for score_idx in range(class_nums):
                    # 记录的是前面一个步骤的每一个节点到当前节点的值
                    temp = []
                    for trellis_idx in range(len(trellis[index - 1])):
                        item = trellis[index - 1][trellis_idx] * score[score_idx] * transition[trellis_idx][score_idx]
                        temp.append(item.item())

                    temp = np.array(temp)
                    # 最大值
                    max_value = np.max(temp)
                    # 最大值下标
                    max_index = np.argmax(temp)
                    # logddd.log(max_value,max_index)
                    # 记录当前节点的前一个节点位置
                    pre_index[index][score_idx] = pre_index[index - 1][max_index] + [score_idx]
                    # logddd.log(pre_index)
                    trellis_cur.append(max_value)
                trellis[index] = np.array(trellis_cur)

        seq_predict_labels = pre_index[-1][np.argmax(trellis[-1])]
        print(seq_predict_labels)
        print(trellis)
        return loss_value, seq_predict_labels, trellis

import json
import csv
if __name__ == '__main__':
    data = """ crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5133809725210832, 'f1': 0.4947030898538106, 'precision': 0.49242282122186204},) Thu Jul  6 13:29:52 2023
 crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5440332844462143, 'f1': 0.5261566327479665, 'precision': 0.5254332046021585},) Thu Jul  6 14:46:33 2023
 crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5445206840193865, 'f1': 0.5331301057533986, 'precision': 0.5378595440030683},) Thu Jul  6 16:09:25 2023
 crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5250873919796281, 'f1': 0.5090662127103831, 'precision': 0.5175470556337521},) Thu Jul  6 17:23:04 2023
 crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5220658678140571, 'f1': 0.49862895276230945, 'precision': 0.4992336586472969},) Thu Jul  6 18:47:18 2023"""
    items = data.split("\n")
    total_res = []
    for item in items:
        item = item.split(":  (")[1].split(",)")[0]
        lines = item.replace("{","").replace("}","").split(",")
        lines[0],lines[1],lines[2] = lines[2],lines[0],lines[1]
        res = []
        for line in lines:
            res.append(str(line.split(": ")[1]))
        total_res.append(res)
    total_res.append([])
    with open("res.csv","a") as f:
            writer = csv.writer(f)
            writer.writerows(total_res)