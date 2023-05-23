import numpy as np
import torch


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


if __name__ == '__main__':
    score = torch.tensor([
        [0.1,0.2,0.7],
        [0.2,0.3,0.5],
        [0.5,0.2,0.3]
    ])
    import torch.nn.functional as F
    score = F.softmax(score,dim=1)
    print(score)
    score = F.softmax(score,dim=1)
    print(score)