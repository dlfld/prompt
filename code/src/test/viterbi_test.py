"""
    下游任务的模型
"""

import logddd
def viterbi_decode_v2(prompts, emmition, transition):
    total_loss = 0
    seq_len, num_labels = len(prompts), len(transition)
    labels = np.arange(num_labels).reshape((1, -1))
    scores = None

    paths = labels

    trellis = None
    for index in range(seq_len):
        loss = 0

        observe = emmition[index]
        print(observe)

        observe = np.array(observe)
        # loss 叠加
        total_loss += loss
        if index == 0:
            # 第一个句子不用和其他的进行比较，直接赋值
            trellis = observe.reshape((1, -1))

            scores = observe
            cur_predict_label_id = np.argmax(observe)
        else:
            M = scores + transition + observe
            scores = np.max(M, axis=0).reshape((-1, 1))
            # shape一下，转为列，方便拼接和找出最大的id(作为预测的标签)
            shape_score = scores.reshape((1, -1))
            # 添加过程矩阵，后面求loss要用
            trellis = np.concatenate([trellis, shape_score], 0)
            # 计算出当前过程的label
            cur_predict_label_id = np.argmax(shape_score)
            idxs = np.argmax(M, axis=0)
            paths = np.concatenate([paths[:, idxs], labels], 0)
        # 如果当前轮次不是最后一轮，那么我们就
    best_path = paths[:, scores.argmax()]
    print("目前的方法")
    print(best_path)

    print(trellis)


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
                        item = trellis[index - 1][trellis_idx] + score[score_idx] + transition[trellis_idx][score_idx]
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


import numpy as np

def get_score(index):
    scores = np.array([
        [0.4,0.6],
        [0.8,0.2],
        [0.7,0.3]    
    ])
    return scores[index]
def viterbi_decode_v3(nodes, trans):
    """
    Viterbi算法求最优路径
    其中 nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    seq_len, num_labels = len(nodes), len(trans)
    labels = np.arange(num_labels).reshape((1, -1))

    scores = nodes[0].reshape((-1, 1))
    logddd.log(nodes[0])
    logddd.log(scores)
    exit(0)
    paths = labels
    logddd.log(paths)
    trills = scores.reshape((1,-1))
 
    for t in range(1, seq_len):
        observe = nodes[t].reshape((1, -1))
        M = scores + trans + observe
        scores = np.max(M, axis=0).reshape((-1, 1))
        shape_score = scores.reshape((1,-1))
        cur_label_id = np.argmax(shape_score)
        trills = np.concatenate([trills,shape_score],0)
        idxs = np.argmax(M, axis=0)
        paths = np.concatenate([paths[:, idxs], labels], 0)


    best_path = paths[:, scores.argmax()]
    print(best_path)
    return best_path


if __name__ == '__main__':
    viterbi = Viterbi_test()
    prompts = np.array([0,0,0])
    scores = np.array([
        [0.4,0.6,],
        [0.8,0.2,],
        [0.7,0.3,]    
    ])
    transition = np.array([
        [0.2,0.8,],
        [0.3,0.7,],
    ])
    # viterbi.viterbi_decode(prompts,scores,transition)
    viterbi_decode_v3(scores,transition)
    # viterbi_decode_v2(prompts, scores, transition)
