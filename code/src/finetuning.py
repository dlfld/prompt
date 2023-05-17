from transformers import pipeline,BertForMaskedLM
import torch
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import AutoModelForMaskedLM
from transformers import TrainingArguments
from data_processing import  get_all_data
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
dataset = load_dataset("csv",data_files="/home/dlf/prompt/dataset.csv")
from transformers import AutoModelForMaskedLM

model_checkpoint = "/home/dlf/prompt/code/model/bert_large_chinese"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    result = tokenizer(examples["text"],return_tensors="pt",padding="max_length",max_length=512)

    # for label in examples["label"]:
    #     print(label)
    #     print(tokenizer.convert_tokens_to_ids(str(label).strip().replace("\n","")))

    result["labels"] = [tokenizer.convert_tokens_to_ids(str(label).strip().replace("\n","")) for label in examples["label"]]
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text","label"]
)
# tokenized_datasets
# print(tokenized_datasets["train"]["labels"])
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.0)
def group_texts(examples):
    result = {
        k: t
        for k, t in examples.items()
    }
    # Create a new labels column
    label = result["labels"].copy()

    result["labels"] = result["input_ids"].copy()
    for index,word in enumerate(result["input_ids"]):
        if word == tokenizer.mask_token_id:
            result["labels"][index] = label[index]
            
    return result
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.00)
import collections
import numpy as np

from transformers import default_data_collator


def whole_word_masking_data_collator(features):
    # for feature in features:
    #     word_ids = feature.pop("word_ids")
 

    #     # Create a map between words and corresponding token indices
    #     mapping = collections.defaultdict(list)
    #     current_word_index = -1
    #     current_word = None
    #     for idx, word_id in enumerate(word_ids):
    #         if word_id is not None:
    #             if word_id != current_word:
    #                 current_word = word_id
    #                 current_word_index += 1
    #             mapping[current_word_index].append(idx)

    #     # Randomly mask words
    #     # mask = np.random.binomial(1, wwm_probability, (len(mapping),))
    #     mask = np.random.binomial(1, 0, (len(feature["input_ids"]),))
    #     # 添加mask位置
    #     for idx,word in enumerate(feature["input_ids"]):
    #         if word == tokenizer.mask_token_id:
    #             mask[idx] = 1

                
        # print(mask)
        # input_ids = feature["input_ids"]
        # labels = feature["labels"]
        # new_labels = [-100] * len(labels)
        # for word_id in np.where(mask)[0]:
        #     word_id = word_id.item()
        #     for idx in mapping[word_id]:
        #         new_labels[idx] = labels[idx]
        #         input_ids[idx] = tokenizer.mask_token_id
                
        # feature["labels"] = new_labels

    return default_data_collator(features)

# samples = [lm_datasets["train"][i] for i in range(2)]
# batch = whole_word_masking_data_collator(samples)

# for chunk in batch["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")
train_size = 10000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)


from transformers import TrainingArguments

batch_size = 32
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
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
import math

# eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

downsampled_dataset = downsampled_dataset.remove_columns(["word_ids","token_type_ids"])
# print(downsampled_dataset)
eval_dataset = downsampled_dataset["test"]

from torch.utils.data import DataLoader
from transformers import default_data_collator

batch_size = 16
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
from tqdm.auto import tqdm
import torch
import math

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss)

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
