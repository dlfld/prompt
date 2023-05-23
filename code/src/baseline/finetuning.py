import datasets
from datasets import load_dataset
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer,BertConfig
import copy
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import math



class Config(object):
    """
        配置类，保存配置文件
    """

    # 测试集位置
    test_dataset_path = ""
    # prompt dataset
    train_dataset_path = "/home/dlf/prompt/dataset.csv"
    # 预训练模型的位置
    model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
    # 训练集大小
    train_size = 60
    # batch_size
    batch_size = 16
    # 学习率
    learning_rate = 2e-5
    # epoch数
    num_train_epochs = 3
    # 句子的最大补齐长度
    sentence_max_len = 128




# 加载模型名字
model_checkpoint = Config.model_checkpoint
# 加载数据
dataset = load_dataset("csv",data_files=Config.train_dataset_path)


# 获取模型配置
model_config = BertConfig.from_pretrained(model_checkpoint)
# 修改配置
model_config.output_hidden_states = True
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint,config = model_config)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ==================数据处理==========================
def tokenize_function(examples):
    result = tokenizer(examples["text"],return_tensors="pt",padding="max_length",max_length=Config.sentence_max_len)
    result["labels"] = [tokenizer.convert_tokens_to_ids(str(label).strip().replace("\n","")) for label in examples["label"]]

    # if tokenizer.is_fast:
    #     result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

    return result

tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text","label"]
)

def group_texts(examples):
    result = {
        k: t
        for k, t in examples.items()
    }

    # Create a new labels column
    # 保存当前列的label
    label = copy.deepcopy(result["labels"])
    # print(label)
    # 复制当前label过去
    labels = copy.deepcopy (result["input_ids"])

    for index,sentence_words in enumerate(result["input_ids"]):
        # 遍历当前句子的每一个word
        for word_index,word in enumerate(sentence_words):
            # 当前word 的id如果是等于mask的id
            if word == tokenizer.mask_token_id:
                # print("进来了")
                # 第index个句子，第word_index个词的id为 = 第index个label
                # result["labels"][index][word_index] = label[index]
                labels[index][word_index] = label[index]
            else:
                labels[index][word_index] = -100
                # print(tokenizer.decode(label[index]))
        # print(tokenizer.decode(labels[index]))

    result["labels"] = labels

    # print(result["labels"] == result["input_ids"])
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# print(lm_datasets)
# ==================数据处理==========================
# 我事先分好训练集和测试集，因此不需要在这个地方分


test_size = int(0.1 * Config.train_size)
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=Config.train_size, test_size=test_size, seed=42
)
# print(downsampled_dataset)
# exit(0)


batch_size = Config.batch_size

# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=Config.learning_rate,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
)


downsampled_dataset = downsampled_dataset.remove_columns(["token_type_ids"])
# eval_dataset = downsampled_dataset["test"]
# print(downsampled_dataset)
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=default_data_collator,
)
# print(train_dataloader)
# exit(0)
# eval_dataloader = DataLoader(
#     eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
# )


optimizer = AdamW(model.parameters(), lr=5e-5)


num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = Config.num_train_epochs * num_update_steps_per_epoch

# 根据epoch的大小来调整学习率
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
# 添加loss 计算

# # 获取自己定义的模型 21128 是词表长度 18是标签类别数
# multi_calss_model = MultiClass(model, 1024,18)
criterion = torch.nn.BCEWithLogitsLoss()
progress_bar = tqdm(range(num_training_steps))
# word_to_labels = ['M', 'DEG', 'VV', 'BP', 'AD', 'PN', 'LC', 'PU', 'NR', 'CD', 'CC', 'JJ', 'VA', 'VC', 'OD', 'NN', 'SP', 'VE']
for epoch in range(Config.num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        # print(batch)
        # print(type(batch))    <class 'dict'>
        # print(batch.keys())  dict_keys(['input_ids', 'attention_mask', 'labels'])
        outputs = model(**batch)
        # multi_calss_model(batch)
        # exit(0)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    # evaluation
    # Evaluation
    model.eval()
    losses = []
    # ⚠️ 这个地方为了测试过拟合用的是 训练集
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss)

    t_losses = torch.tensor(losses)
    t_losses = t_losses[: Config.train_size]

    try:
        perplexity = math.exp(torch.mean(t_losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
    datas = [
        "在句子“脉细弱”中，词语“脉”的前文如果是由“[CLS]”词性的词语“[CLS]”来修饰，那么词语“脉”的词性是“[MASK]”",
        "在句子“脉细弱”中，词语“细”的前文如果是由“NR”词性的词语“脉”来修饰，那么词语“细”的词性是“[MASK]”",
        "在句子“脉细弱”中，词语“弱”的前文如果是由“VA”词性的词语“细”来修饰，那么词语“弱”的词性是“[MASK]”",
        "在句子“脉软”中，词语“脉”的前文如果是由“[CLS]”词性的词语“[CLS]”来修饰，那么词语“脉”的词性是“[MASK]”",
        "在句子“脉软”中，词语“软”的前文如果是由“NR”词性的词语“软”来修饰，那么词语“软”的词性是“[MASK]”",
        "在句子“脉细”中，词语“脉”的前文如果是由“[CLS]”词性的词语“[CLS]”来修饰，那么词语“脉”的词性是“[MASK]”,",
        "在句子“脉细”中，词语“细”的前文如果是由“NR”词性的词语“细”来修饰，那么词语“细”的词性是“[MASK]”",
    ]
    with torch.no_grad():
        for item in datas:
            inputs = tokenizer(item, return_tensors="pt")
            print(inputs)
            token_logits = model(**inputs).logits
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
            mask_token_logits = token_logits[0, mask_token_index, :]
            top_5_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
            for token in top_5_tokens:
                print(f"'>>> {item.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
    # test_model(model=model,dataloader=train_dataloader,tokenizer=tokenizer,epoch=epoch,Config=Config)