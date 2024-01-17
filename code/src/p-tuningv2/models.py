import torch
import torch.nn.functional as F
from torch import nn
import logddd
from model_params import Config
from prefix_encoder import PrefixEncoder

"""
    下游任务的模型
"""
import numpy as np


class SequenceLabeling(nn.Module):
    def __init__(self, bert_model, hidden_size, class_nums, tokenizer, bert_config):
        """
            @param bert_model: 预训练模型
            @param hidden_size: 隐藏层大小
            @param class_nums: 类别数
            @param tokenizer:tokenizer
        """
        super(SequenceLabeling, self).__init__()
        self.bert_config = bert_config
        # bert 模型
        self.bert = bert_model.to(Config.device)
    
        self.model_type = type(self.bert).__name__

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
        # ----------------------------p-tuning-v2---------------------------------

        self.dropout = torch.nn.Dropout(Config.hidden_dropout_prob)
        # 这个使用一个线性层将结果计算为label数量
        self.classifier = torch.nn.Linear(bert_config.hidden_size, self.class_nums)

        # 冻结bert的参数，p-tuning-v2是需要冻结bert参数的
        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = Config.pre_seq_len
        self.n_layer = bert_config.num_hidden_layers
        self.n_head = bert_config.num_attention_heads
        self.n_embd = bert_config.hidden_size // bert_config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_hidden_size = Config.prefix_hidden_size

        self.prefix_encoder = PrefixEncoder(bert_config, self.pre_seq_len, self.prefix_hidden_size)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    # def get_prompt(self, batch_size):
    #     # 随机生成一个一纬张量
    #     prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
    #     # 使用nn.embedding获取其词向量
    #     prompts = self.prefix_encoder(prefix_tokens)
    #     return prompts

    def forward(self, datas):
        # 取出一条数据,也就是一组prompt,将这一组prompt进行维特比计算
        # 所有predict的label
        total_predict_labels = []
        # 所有的score
        total_scores = []
        # 每一条数据中bert的loss求和
        total_loss = 0
        # 遍历每一个句子生成的prompts
        for index, data in enumerate(datas):
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

        # 到这一步就生成了一个nums([T])的embedding，下一步就是要去替换每一个句子中的embedding
        # 如何找到原来prompt句子中的每一个[T]对应的位置？
        prompt = {
            k: torch.tensor(v).to(Config.device)
            for k, v in prompt.items()
        }
        input_ids = prompt["input_ids"]
        attention_mask = prompt["attention_mask"]
        # logddd.log(attention_mask)
        # logddd.log(attention_mask.shape)
        # token_type_ids = prompt["token_type_ids"]
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        # logddd.log(prefix_attention_mask.shape)
    

        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if "Bart" in self.model_type:
            apd = attention_mask.shape[1] - input_ids.shape[1]
            input_ids = torch.cat((input_ids,torch.zeros(1,apd,dtype=torch.long).to(device = Config.device)),dim = 1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            past_key_values=past_key_values,
        )
        if "Bart" in self.model_type:
           pooled_output = outputs[0]
        else:
           pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        mask_embedding = logits

        if "Bart" in self.model_type:
            # 获取到mask维度的label
            # 遍历每一个句子 抽取出被mask位置的隐藏向量, 也就是抽取出mask
            for label_index, sentences in enumerate(prompt["input_ids"]):
                # 遍历句子中的每一词,
                for word_index, val in enumerate(sentences):
                    if val == self.tokenizer.mask_token_id:
                        # predict_labels.append(out_fc[label_index][word_index].tolist())
                        mask_embedding = logits[:, word_index, :]
                        break
        # logddd.log(mask_embedding.shape)
        return [mask_embedding.tolist()], 0

    def viterbi_decode_v3(self, prompts):
        """
        Viterbi算法求最优路径
        其中 nodes.shape=[seq_len, num_labels],
            trans.shape=[num_labels, num_labels].
        """

        # for item in prompts["input_ids"]:
        #     logddd.log(self.tokenizer.convert_ids_to_tokens(item))
        seq_len, num_labels = len(prompts["input_ids"]), len(self.transition_params)
        labels = torch.arange(num_labels).view((1, -1)).to(device=Config.device)
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
            # logit = template_logit[0][0]
            logit = np.array(template_logit[0][0])
            logit = torch.from_numpy(logit).to(Config.device)
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
                trellis = observe.reshape((1, -1))
                scores = observe
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
