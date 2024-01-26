import torch
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PromptEncoder, PromptEncoderConfig
from peft.utils import TaskType

# 加载分词器和模型
path = ""
tokenizer = BertTokenizer.from_pretrained('/Users/dailinfeng/Desktop/prompt/code/model/bert_large_chinese')
model = BertForSequenceClassification.from_pretrained('/Users/dailinfeng/Desktop/prompt/code/model/bert_large_chinese', num_labels=2)  # 假设有两个类别

# 配置 P-Tuning v2
peft_config = PromptEncoderConfig(
    task_type=TaskType.SEQ_CLS,  # 任务类型
    num_virtual_tokens=20,  # 虚拟令牌数量
    token_dim=768,  # BERT 嵌入维度
    encoder_reparameterization_type="LSTM",  # 编码器重参数化类型
    encoder_hidden_size=768,
    encoder_num_layers=2,
    encoder_dropout=0.1
)

prompt_encoder = PromptEncoder(peft_config)

# 准备输入文本
text = "Your example sentence here.[MASK]"
inputs = tokenizer(text, return_tensors="pt")

# 将 Prompt Encoder 应用于输入
virtual_tokens = prompt_encoder(inputs['input_ids'].shape[0])
inputs['input_ids'] = torch.cat([virtual_tokens, inputs['input_ids']], dim=1)

# 预测
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

# 输出分类结果
print(predictions)