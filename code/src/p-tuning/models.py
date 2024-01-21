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
        self.dropout = torch.nn.Dropout(0.2)
        # self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        # ----------------------p-tuning------------------------
        # 是否更新bert的参数
        self.update_bert = True
        for param in self.bert.parameters():
            param.requires_grad = True

        self.T = tokenizer.convert_tokens_to_ids("[T]")
        self.hidden_size = Config.embed_size
        # 当前提示模板中[T]的数量
        self.prompt_length = Config.prompt_length
        # prompt_length 连续提示的数量
        self.prompt_embeddings = torch.nn.Embedding(Config.prompt_length, Config.embed_size)
        if Config.prompt_encoder_type == "lstm":
            self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.hidden_size))

        elif Config.prompt_encoder_type == "mlp":
            self.mlp = nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size))
        # -------------------------------------------------------------
        elif Config.prompt_encoder_type == "gru":
            self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size,num_layers=2)
            self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, self.hidden_size))



    def forward(self, datas):
        # input_ids = datas[1]["input_ids"].to(Config.device)
        # logddd.log(input_ids.shape)
        # # logddd.log(self.bert)
        # word_embedding = self.bert.bert.embeddings.word_embeddings(input_ids)
        # logddd.log(word_embedding.shape)
        # replace_embeds = self.prompt_embeddings(
        #     torch.LongTensor(list(range(self.prompt_length))).cuda()
        # )
        # logddd.log(replace_embeds.shape)
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
        input_ids = prompt["input_ids"]
        input_ids = torch.tensor(input_ids[0])
       
        # [T]标签所在的位置
        t_locals = torch.where(input_ids == self.T)
        # logddd.log(t_locals)
        prompt = {
            k: torch.tensor(v).to(Config.device)
            for k, v in prompt.items()
        }
        input_ids = prompt["input_ids"]

        # shape 1 256 1024
        # 一句话的embedding   一个prompt的
        
        raw_embeds = self.bert.bert.embeddings.word_embeddings(input_ids)
        # raw_embeds = self.bert.model.encoder.embed_tokens(input_ids)
        replace_embeds = self.prompt_embeddings(
            torch.LongTensor(list(range(self.prompt_length))).to(device=Config.device)
        )
        # logddd.log(replace_embeds.shape)
        # [batch_size, prompt_length, embed_size]  1 nums([T]) 1024
        replace_embeds = replace_embeds.unsqueeze(0) 
        # logddd.log(replace_embeds.shape)
        
        if Config.prompt_encoder_type == "lstm":
            replace_embeds = self.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
            replace_embeds = self.mlp_head(replace_embeds).squeeze()

        elif Config.prompt_encoder_type == "mlp":
            self.mlp(replace_embeds)
        elif Config.prompt_encoder_type == "gru":
            replace_embeds = self.gru(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
            replace_embeds = self.mlp_head(replace_embeds).squeeze()
     
            # 依次替换
        # replace_embeds 6 * 1024
        # 下面就要依次替换
        index = 0
        for i in range(raw_embeds.shape[0]):
            for j in range(raw_embeds.shape[1]):
                if j in t_locals[0].tolist():
                    raw_embeds[i][j] = replace_embeds[index]
                    index += 1

        # logddd.log(prompt.keys())
        # 替换完成，使用经过LSTM head的embedding替换[T] 伪提示的embedding
        inputs = {
            'inputs_embeds': raw_embeds, 
            'attention_mask': prompt['attention_mask'],
        }
        if 'labels' in prompt.keys():
            inputs['labels'] = prompt['labels'] 

        if self.update_bert:
            outputs = self.bert(**inputs)
            out_fc = outputs.logits
            loss = outputs.loss
            if loss.requires_grad:
                loss.backward(retain_graph=True)
        else:
            self.bert.eval()
            with torch.no_grad():
        # # 输入bert预训练
                outputs = self.bert(**inputs)
                out_fc = outputs.logits
            loss = outputs.loss


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
        predict_score = [mask_embedding[:, 1:1 + Config.class_nums].tolist()]

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
