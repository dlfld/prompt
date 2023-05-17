from transformers import pipeline,BertForMaskedLM
import torch
from transformers import BertTokenizer
from transformers import BertForMaskedLM

if __name__ == '__main__':
    datas = ['在句子"脉细弦"中，词语"脉"的前文如果是由"[CLS]"词性的词语"[CLS]"来修饰，那么词语"脉"的词性是"[MASK]"→ NR', '在句子"脉细弦"中，词语"细"的前文如果是由"NR"词性的词语"脉"来修饰，那么词语"细"的词性是"[MASK]"→ VA', '在句子"脉细弦"中，词语"弦"的前文如果是由"VA"词性的词语"细"来修饰，那么词语"弦"的词性是"[MASK]"→ VA']

    # model = BertForMaskedLM.from_pretrained("/home/dlf/prompt/code/model/bert_large_chinese")
    # model = pipeline("fill-mask", model="/home/dlf/prompt/code/model/bert_large_chinese")
    # for data in datas:
    #     res = model(data.split("→")[0])
    #     for item in res:
    #         print(item)
    #     print()


    model_checkpoint =  "/home/dlf/prompt/code/model/bert_large_chinese"
    from transformers import AutoModelForMaskedLM

    # model_checkpoint = "distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    import torch
    # text = "This is a great [MASK]."
    text = "在句子“脉细”中，词语“细”的前文如果是由“NR”词性的词语“脉”来修饰，那么词语“细”的词性是“[MASK]”"
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)
    # exit(0)
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")


    # model =  "/home/dlf/prompt/code/model/bert_large_chinese"
    # # # 获取tokenizer
    # bert_tokenzier = BertTokenizer.from_pretrained(model)
    # # # 加载bert mask 预训练模型
    # maskbert = BertForMaskedLM.from_pretrained(model)
    # for data in datas:
    #     text = data.split("→")[0]
    #     text_tokens = bert_tokenzier(text, add_special_tokens=True,padding=True, return_tensors='pt')
    #     maskbert_outputs = maskbert(**text_tokens, return_dict=True,output_hidden_states=True)
    #     logits = maskbert_outputs.logits
    #     pred = torch.argmax(logits, dim=-1)
    #     pred = pred.data.cpu().numpy().tolist()[0]
    #     pred_tokens = bert_tokenzier.decode(pred)
    #     print(pred_tokens)