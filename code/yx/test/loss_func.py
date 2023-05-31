import logddd
import torch
from torch import nn
from transformers import BertConfig, AutoModelForMaskedLM, AutoTokenizer
import sys

def test1():
    loss = nn.CrossEntropyLoss(ignore_index=-100)
    input = torch.randn(3, 5, requires_grad=True).softmax(dim=1)
    print(input)
    target = torch.tensor([
        [-100, -100, -100, -100, 10],
        [18, -100, -100, -100, -100],
        [-100, 16, -100, -100, -100]
    ], dtype=float)
    print(target)
    output = loss(input, target)
    print(output)

def test2():
    loss = nn.CrossEntropyLoss(ignore_index=-100)
    # input = torch.randn(1, 5, requires_grad=True).softmax(dim=1)
    input = torch.tensor([[0.2330, 0.2873, 0.3101, 0.0764, 0.0932],[0.2330, 0.2873, 0.3101, 0.0764, 0.0932],[0.2330, 0.2873, 0.3101, 0.0764, 0.0932]]).softmax(dim=1)
    print(input)
    target = torch.tensor([[-100, -100, -100, -100, 10],[-100, -100, -100, -100, 10],[-100, -100, -100, -100, 10]], dtype=float)
    print(target)
    output = loss(input, target)
    print(output)
def test3():
    import torch

    # 示例张量
    tensor = torch.tensor([-100, 1, -100,  -100, -100,  -100])

    # 找到不等于 -100 的值
    non_negative_values = tensor[tensor != -100]

    print(non_negative_values)
def test4():
    import torch

    # 示例值
    values = torch.tensor([ 3])

    # 生成One-Hot向量
    one_hot = torch.eye(18)[values]

    print(one_hot)

def test5():
    model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': ["[PLB]"]})
    logddd.log(tokenizer.convert_tokens_to_ids("[PLB]"))
    logddd.log(tokenizer.convert_ids_to_tokens(21))
    item = tokenizer("在句子“脉软”中，词语“软”的前文如果是由“[PLB]”词性的词语“脉”来修饰，那么词语“软”的词性是“[MASK]”→ VA")
    logddd.log(item)
    for a in item["input_ids"]:
        print(tokenizer.convert_ids_to_tokens(a))

def crf_test():
    import torch
    from torchcrf import CRF
    num_tags = 5  # number of tags is 5
    model = CRF(num_tags,batch_first=True)
    seq_length = 3
    batch_size = 2
    emissions = torch.randn(batch_size, seq_length, num_tags)
    tags = torch.tensor([[0, 1], [2, 4], [3, 1]], dtype=torch.long)  # (seq_length, batch_size)
    model(emissions, tags)

if __name__ == '__main__':
    print(type(torch.tensor([1])) == torch.Tensor)