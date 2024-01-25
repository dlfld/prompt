import logddd
import torch
import torch.nn.functional as F
from torch import nn

from model_params import Config

"""
    下游任务的模型
"""
import numpy as np
from torch.optim import AdamW


class SequenceLabeling(nn.Module):
    def get_label_embeddings(self):
        """
            获取当前数据集所有标签的embedding
        """
        # 所有的标签
        labels = Config.special_labels[1:]
        label_tokens = self.tokenizer(labels, return_tensors='pt', padding="max_length",
                                      max_length=Config.sentence_max_len)
        # logddd.log(label_tokens)
        label_tokens = {
            k: torch.tensor(v).to(Config.device)
            for k, v in label_tokens.items()
        }
        self.bert.eval()
        with torch.no_grad():
            outputs = self.bert(**label_tokens, output_hidden_states=True)
            # label_embeddings = outputs.logits
            logddd.log(outputs.logits.shape)
            label_embeddings = outputs.hidden_states[-1]
        self.bert.train()
        # 获取中间一层的embedding,并做转制
        label_embeddings = label_embeddings[:, 1, :]

        # exit(0)
        return label_embeddings

    def get_logit(self, h_mask):
        """
            获取分母
        """
        items = []
        for item in self.labels_embeddings:
            #        "获取每一个标签的embedding"
            item = torch.unsqueeze(item, 0)
            temp = torch.mm(item, torch.transpose(h_mask, 0, 1))
            # temp = F.cosine_similarity(item, h_mask)
            cur = torch.exp(temp)
            items.append(cur)
        # 堆叠成一个新的tensor
        norm_items = torch.stack(items)
        deno_sum = torch.sum(norm_items)
        res = []
        # 遍历每一个标签，取出每一个标签的概率
        for item in norm_items:
            # 添加每一个标签的预测概率
            res.append(item / deno_sum)
        res = torch.stack(res).squeeze().unsqueeze(0)

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
        # for index,param in enumerate(self.bert.parameters()):
        #     if index %2==0:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # 当前所有标签的embedding
        # self.labels_embeddings = self.get_label_embeddings()
        # self.dropout = torch.nn.Dropout(0.2)
        # self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        # self.optimizer = AdamW(self.bert.parameters(), lr=Config.learning_rate)
        # self.optimizer = AdamW(self.bert.parameters(), lr=Config.learning_rate)
    #
    def forward(self, datas):
        # 取出一条数据,也就是一组prompt,将这一组prompt进行维特比计算
        # 所有predict的label
        total_predict_labels = []
        # 所有的score
        total_scores = []
        # 每一条数据中bert的loss求和
        total_loss = 0
        # 遍历每一个句子生成的prompts
        # logddd.log(self.transition_params.tolist())
        # logddd.log(len(datas))
        for index, data in enumerate(datas):
            # self.viterbi_decode_v2(data)
            scores, seq_predict_labels, loss = self.viterbi_decode_v3(data)
            total_predict_labels.append(seq_predict_labels)
            total_scores.append(scores)
            total_loss += loss

        # self.transition_params.retain_grad()
        # logddd.log(self.transition_params.grad)
        # logddd.log(self.transition_params)
        # self.transition_params.backward(retain_graph=True)
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
        outputs = self.bert(**prompt,output_hidden_states=True)
        # outputs = self.bert(**prompt)
        # replace_embeds = model.prompt_embeddings(
        #     torch.LongTensor(list(range(model.prompt_length))).cuda()
        # )
        out_fc = outputs.logits
        output_hidden_states = outputs.hidden_states
        # logddd.log(output_hidden_states.shape)
        loss = outputs.loss
        if loss.requires_grad:
            loss.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()

        #     # loss.backward()

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
                    # mask_embedding = output_hidden_states[:,word_index,:]
                    break

        # 获取指定位置的数据，之前的方式，截取
        # logddd.log(mask_embedding.shape)
        # mask_embedding = mask_embedding[:, 1:1 + Config.class_nums]
        # exit(0)
        # predict_score = [score[1:1 + Config.class_nums] for score in predict_labels]

        predict_score = [mask_embedding[:, 1:1 + Config.class_nums].tolist()]

        del prompt, outputs, out_fc
        return predict_score, loss.item()
        # return predict_score, loss

    def viterbi_decode_v4(self, prompts):
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
                trills = scores.reshape((1, -1))
                cur_predict_label_id = np.argmax(scores) + 1
            else:
                observe = logit.reshape((1, -1))
                M = scores + self.transition_params.cpu().detach().numpy() + observe
                scores = np.max(M, axis=0).reshape((-1, 1))

                shape_score = scores.reshape((1, -1))

                cur_predict_label_id = np.argmax(shape_score) + 1
                trills = np.concatenate([trills, shape_score], 0)
                idxs = np.argmax(M, axis=0)
                paths = np.concatenate([paths[:, idxs], labels], 0)

            if index != seq_len - 1:
                next_prompt = prompts["input_ids"][index + 1]
                next_prompt = torch.tensor([x if x != self.PLB else cur_predict_label_id for x in next_prompt])
                prompts["input_ids"][index + 1] = next_prompt

        best_path = paths[:, scores.argmax()]

        return F.softmax(torch.tensor(trills)), best_path, total_loss / seq_len

    def viterbi_decode_v3(self, prompts):
        """
        Viterbi算法求最优路径
        其中 nodes.shape=[seq_len, num_labels],
            trans.shape=[num_labels, num_labels].
        """

        # for item in prompts["input_ids"]:
        #     logddd.log(self.tokenizer.convert_ids_to_tokens(item))
        seq_len, num_labels = len(prompts["input_ids"]), len(self.transition_params)
        labels = torch.arange(num_labels).view((1, -1)).to(Config.device)
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
            logit = torch.from_numpy(logit).to(Config.device)
            # logit = template_logit[0][0]
            total_loss += loss
            if index == 0:
                scores = logit.view(-1, 1)
                trills = scores.view(1, -1)
                cur_predict_label_id = torch.argmax(scores) + 1
            else:
                observe = logit.view(1, -1)
                M = scores + self.transition_params + observe
                scores = torch.max(M, dim=0)[0].view(-1, 1)
                shape_score = scores.view(1, -1)
                cur_predict_label_id = torch.argmax(shape_score) + 1
                trills = torch.cat((trills, shape_score), dim=0)
                idxs = torch.argmax(M, dim=0)
                paths = torch.cat((paths[:, idxs], labels), dim=0)

            if index != seq_len - 1:
                next_prompt = prompts["input_ids"][index + 1]
                next_prompt = torch.tensor([x if x != self.PLB else cur_predict_label_id for x in next_prompt])
                prompts["input_ids"][index + 1] = next_prompt

        best_path = paths[:, scores.argmax()]
        return F.softmax(trills), best_path, total_loss / seq_len

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
                cur_predict_label_id = np.argmax(observe) + 1
            else:
                M = scores + self.transition_params.cpu().detach().numpy() + observe
                scores = np.max(M, axis=0).reshape((-1, 1))
                # shape一下，转为列，方便拼接和找出最大的id(作为预测的标签)
                shape_score = scores.reshape((1, -1))
                # 添加过程矩阵，后面求loss要用
                trellis = np.concatenate([trellis, shape_score], 0)
                # 计算出当前过程的label
                cur_predict_label_id = np.argmax(shape_score) + 1
                idxs = np.argmax(M, axis=0)
                paths = np.concatenate([paths[:, idxs], labels], 0)
            # 如果当前轮次不是最后一轮，那么我们就
            if index != seq_len - 1:
                next_prompt = prompts["input_ids"][index + 1]
                next_prompt = torch.tensor([x if x != self.PLB else cur_predict_label_id for x in next_prompt])
                prompts["input_ids"][index + 1] = next_prompt

        best_path = paths[:, scores.argmax()]
        # 这儿返回去的是所有的每一句话的平均loss
        return F.softmax(torch.tensor(trellis)), best_path, total_loss / seq_len
        # return torch.tensor(trellis), best_path, total_loss / seq_len
