import logddd
import torch
import torch.nn.functional as F
from torch import nn

from model_params import Config

"""
    下游任务的模型
"""
import numpy as np


class SequenceLabeling(nn.Module):
    def get_label_embeddings(self):
        """
            获取当前数据集所有标签的embedding
        """
        # 所有的标签
        labels = Config.special_labels[1:]
        label_tokens = self.tokenizer(labels, return_tensors='pt', padding="max_length",
                                      max_length=Config.sentence_max_len)
        label_tokens = {
            k: torch.tensor(v).to(Config.device)
            for k, v in label_tokens.items()
        }
        self.bert.eval()
        label_embeddings = self.bert(**label_tokens).logits
        self.bert.train()
        # 获取中间一层的embedding,并做转制
        label_embeddings = label_embeddings[:, 1, :]
        return label_embeddings

    def get_logit(self, h_mask):
        """
            获取分母
        """
        items = []
        for item in self.labels_embeddings:
            #        "获取每一个标签的embedding"
            item = torch.unsqueeze(item, 0)
            temp = torch.mm(item, torch.transpose(h_mask, 0, 1)) / 1e6
            # temp = F.cosine_similarity(item, h_mask)
            cur = torch.exp(temp)
            logddd.log(temp)

            items.append(cur)
        exit(0)
        # 堆叠成一个新的tensor
        norm_items = torch.stack(items)

        deno_sum = torch.sum(norm_items)
        res = []
        # 遍历每一个标签，取出每一个标签的概率
        for item in norm_items:
            # logddd.log(item)
            # 添加每一个标签的预测概率
            res.append(item / deno_sum)
        res = torch.stack(res)

        ans = res.tolist()
        return ans

    def __init__(self, bert_model, hidden_size, class_nums, tokenizer):
        """
            @param bert_model: 预训练模型
            @param hidden_size: 隐藏层大小
            @param class_nums: 类别数
            @param tokenizer:tokenizer
        """
        super(SequenceLabeling, self).__init__()
        # bert 模型
        self.bert = bert_model.to(Config.device)
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
        # 当前所有标签的embedding
        # self.labels_embeddings = self.get_label_embeddings()

    def forward(self, datas):
        # 取出一条数据,也就是一组prompt,将这一组prompt进行维特比计算
        # 所有predict的label
        total_predict_labels = []
        # 所有的score
        total_scores = []
        # 每一条数据中bert的loss求和
        total_loss = 0
        # 遍历每一个句子生成的prompts
        for index,data in enumerate(datas):
            # self.viterbi_decode_v2(data)
            scores, seq_predict_labels, loss = self.viterbi_decode_v3(data)
            total_predict_labels.append(seq_predict_labels)
            total_scores.append(scores)
            total_loss += loss

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
        prompt = {
            k: torch.tensor(v).to(Config.device)
            for k, v in prompt.items()
        }
        # 输入bert预训练
        outputs = self.bert(**prompt)

        out_fc = outputs.logits

        loss = outputs.loss
        if loss.requires_grad:
            loss.backward()

        mask_embedding = None
        # 获取到mask维度的label
        predict_labels = []
        # 遍历每一个句子 抽取出被mask位置的隐藏向量, 也就是抽取出mask
        for label_index, sentences in enumerate(prompt["input_ids"]):
            # 遍历句子中的每一词,
            for word_index, val in enumerate(sentences):
                if val == self.tokenizer.mask_token_id:
                    # predict_labels.append(out_fc[label_index][word_index].tolist())
                    mask_embedding = out_fc[:, word_index, :]
                    break

        # 获取指定位置的数据，之前的方式，截取
        # logddd.log(mask_embedding.shape)
        # mask_embedding = mask_embedding[:, 1:1 + Config.class_nums]
        # exit(0)
        # predict_score = [score[1:1 + Config.class_nums] for score in predict_labels]
        predict_score = [mask_embedding[:, 1:1 + Config.class_nums].tolist()]
        # logddd.log(predict_score)
        # predict_score = [self.get_logit(mask_embedding)]

        del prompt, outputs, out_fc
        return predict_score, loss.item()

    def viterbi_decode_v3(self, prompts):
        """
        Viterbi算法求最优路径
        其中 nodes.shape=[seq_len, num_labels],
            trans.shape=[num_labels, num_labels].
        """

        # for item in prompts["input_ids"]:
        #     logddd.log(self.tokenizer.convert_ids_to_tokens(item))
        seq_len, num_labels = len(prompts["input_ids"]), len(self.transition_params)
        labels = np.arange(num_labels).reshape((1, -1))
        paths = labels
        trills = None
        scores = None
        total_loss = 0
        for index in range(seq_len):
            cur_data = {
                k: [v[index].tolist()]
                for k, v in prompts.items()
            }
            template_logit, loss = self.get_score(cur_data)
            logit = np.array(template_logit[0][0])
            total_loss += loss
            if index == 0:
                scores = logit.reshape((-1, 1))
                trills = scores.reshape((1,-1))
                cur_predict_label_id = np.argmax(scores)
            else:
                observe = logit.reshape((1,-1))
                M = scores + self.transition_params.cpu().detach().numpy()  + observe
                scores = np.max(M, axis=0).reshape((-1, 1))
                shape_score = scores.reshape((1,-1))
                cur_predict_label_id = np.argmax(shape_score)
                trills = np.concatenate([trills, shape_score], 0)
                idxs = np.argmax(M, axis=0)
                paths = np.concatenate([paths[:, idxs], labels], 0)

            if index != seq_len - 1:
                next_prompt = prompts["input_ids"][index + 1]
                next_prompt = torch.tensor([x if x != self.PLB else cur_predict_label_id for x in next_prompt])
                prompts["input_ids"][index + 1] = next_prompt

        best_path = paths[:, scores.argmax()]
        return F.softmax(torch.tensor(trills)), best_path, total_loss / seq_len

    def viterbi_decode_v2(self, prompts):
        total_loss = 0
        seq_len, num_labels = len(prompts["input_ids"]), len(self.transition_params)
        labels = np.arange(num_labels).reshape((1, -1))
        scores = None
        paths = labels

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
                shape_score = scores.reshape((1, -1))
                # 添加过程矩阵，后面求loss要用
                trellis = np.concatenate([trellis, shape_score], 0)
                # 计算出当前过程的label
                cur_predict_label_id = np.argmax(shape_score)
                idxs = np.argmax(M, axis=0)
                paths = np.concatenate([paths[:, idxs], labels], 0)
            # 如果当前轮次不是最后一轮，那么我们就
            if index != seq_len - 1:
                next_prompt = prompts["input_ids"][index + 1]
                next_prompt = torch.tensor([x if x != self.PLB else cur_predict_label_id for x in next_prompt])
                prompts["input_ids"][index + 1] = next_prompt

        best_path = paths[:, scores.argmax()]
        # 这儿返回去的是所有的每一句话的平均loss
        logddd.log(F.softmax(torch.tensor(trellis)).shape)
        logddd.log(best_path)
        logddd.log(total_loss / seq_len)
        return F.softmax(torch.tensor(trellis)), best_path, total_loss / seq_len
        # return torch.tensor(trellis), best_path, total_loss / seq_len
