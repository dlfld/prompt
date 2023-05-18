from transformers import pipeline,BertForMaskedLM
import torch
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import AutoModelForMaskedLM
from transformers import TrainingArguments
# from data_processing import  get_all_data
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
# from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import math
from transformers import AutoTokenizer
from datasets import load_dataset,DatasetDict,Dataset
from datasets import load_dataset
from models import MultiClass
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import copy
from transformers import DataCollatorForLanguageModeling
import collections
import numpy as np

from transformers import default_data_collator
from transformers import TrainingArguments
from transformers import Trainer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import math

dataset = load_dataset("csv",data_files="/home/dlf/prompt/dataset.csv")

model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    result = tokenizer(examples["text"],return_tensors="pt",padding="max_length",max_length=128)

    # for label in examples["label"]:
    #     print(label)
    #     print(tokenizer.convert_tokens_to_ids(str(label).strip().replace("\n","")))

    result["labels"] = [tokenizer.convert_tokens_to_ids(str(label).strip().replace("\n","")) for label in examples["label"]]
    # print(result["labels"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    
    return result


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text","label"]
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.0)


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
# exit(0)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.00)


def whole_word_masking_data_collator(features):

    return default_data_collator(features)

# samples = [lm_datasets["train"][i] for i in range(2)]
# batch = whole_word_masking_data_collator(samples)

# for chunk in batch["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")
train_size = 60
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)



batch_size = 16
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=downsampled_dataset["train"],
#     eval_dataset=downsampled_dataset["test"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )

# eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

downsampled_dataset = downsampled_dataset.remove_columns(["word_ids","token_type_ids"])
eval_dataset = downsampled_dataset["test"]

train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=default_data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# 获取自己定义的模型 21128 是词表长度 18是标签类别数
multi_calss_model = MultiClass(model, 21128,18)
criterion = torch.nn.BCEWithLogitsLoss()
progress_bar = tqdm(range(num_training_steps))
import logddd
word_to_labels = ['M', 'DEG', 'VV', 'BP', 'AD', 'PN', 'LC', 'PU', 'NR', 'CD', 'CC', 'JJ', 'VA', 'VC', 'OD', 'NN', 'SP', 'VE']
for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        # logddd.log(tokenizer.decode(sentence[0].tolist()))
        # outputs = multi_calss_model(batch)
        # loss = criterion(outputs,batch["labels"])

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()

    losses = []
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss)

    t_losses = torch.tensor(losses)
    t_losses = t_losses[: train_size]
    try:
        perplexity = math.exp(torch.mean(t_losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    datas = [
        ['在句子“脉细弱”中，词语“脉”的前文如果是由“[CLS]”词性的词语“[CLS]”来修饰，那么词语“脉”的词性是“[MASK]”', 'NR'],
        ['在句子“脉细弱”中，词语“细”的前文如果是由“NR”词性的词语“脉”来修饰，那么词语“细”的词性是“[MASK]”', 'VA'],
        ['在句子“脉细弱”中，词语“弱”的前文如果是由“VA”词性的词语“细”来修饰，那么词语“弱”的词性是“[MASK]”', 'VA'],

        ['在句子“脉软”中，词语“脉”的前文如果是由“[CLS]”词性的词语“[CLS]”来修饰，那么词语“脉”的词性是“[MASK]”', 'NR'],
        ['在句子“脉软”中，词语“软”的前文如果是由“NR”词性的词语“脉”来修饰，那么词语“软”的词性是“[MASK]”', 'VA'],

        ['在句子“脉细”中，词语“脉”的前文如果是由“[CLS]”词性的词语“[CLS]”来修饰，那么词语“脉”的词性是“[MASK]”', 'NR'],
        ['在句子“脉细”中，词语“细”的前文如果是由“NR”词性的词语“脉”来修饰，那么词语“细”的词性是“[MASK]”', 'VA'],

    ]
    with torch.no_grad():
        for item in datas:
            # print(item)
            inputs = tokenizer(item[0], return_tensors="pt")
    
            token_logits = model(**inputs).logits
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
            mask_token_logits = token_logits[0, mask_token_index, :]
            top_5_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
            for token in top_5_tokens:
                print(f"'>>> {item[0].replace(tokenizer.mask_token, tokenizer.decode([token]))}' -> 正确值为： {item[1]}" )
