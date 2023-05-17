from transformers import pipeline,BertForMaskedLM
import torch
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import TrainingArguments
from data_processing import  get_all_data


datas = ['在句子"脉细弦"中，词语"脉"的前文如果是由"[CLS]"词性的词语"[CLS]"来修饰，那么词语"脉"的词性是"[MASK]"→ NR', '在句子"脉细弦"中，词语"细"的前文如果是由"NR"词性的词语"脉"来修饰，那么词语"细"的词性是"[MASK]"→ VA', '在句子"脉细弦"中，词语"弦"的前文如果是由"VA"词性的词语"细"来修饰，那么词语"弦"的词性是"[MASK]"→ VA']

model_checkpoint =  "/home/dlf/prompt/code/model/bert_large_chinese"
# model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# for item in datas:
#     text = item.split("→")[0]
#     inputs = tokenizer(text, return_tensors="pt")
#     token_logits = model(**inputs).logits
#     # Find the location of [MASK] and extract its logits
#     mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
#     mask_token_logits = token_logits[0, mask_token_index, :]
#     # Pick the [MASK] candidates with the highest logits
#     top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

#     for token in top_5_tokens:
#         print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

import logddd
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")
print(imdb_dataset)
import numpy as np
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)

chunk_size = 128
# 将块的大小调整为统一的大小
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.1)

samples = [lm_datasets["train"][i] for i in range(2)]

for sample in samples:
    _ = sample.pop("word_ids")
    print(sample)

print("chunk")
for chunk in data_collator(samples)["input_ids"]:
    # [CLS] bromwell [MASK] is a cartoon comedy. it ran at the same [MASK] as some other [MASK] about school life, [MASK] as " teachers ". [MASK] [MASK] [MASK] in the teaching [MASK] lead [MASK] to believe that bromwell high\'[MASK] satire is much closer to reality than is " teachers ". the scramble [MASK] [MASK] financially, the [MASK]ful students whogn [MASK] right through [MASK] pathetic teachers\'pomp, the pettiness of the whole situation, distinction remind me of the schools i knew and their students. when i saw [MASK] episode in [MASK] a student repeatedly tried to burn down the school, [MASK] immediately recalled. [MASK]...'
    # 获取到的是加了[]
    print(chunk)
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
import collections
import numpy as np
 
from transformers import default_data_collator
 
wwm_probability = 0.2
 
 
def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")
 
        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)
 
        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
 
    return default_data_collator(features)

samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

print("batch")
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
print(batch.keys())
print(batch['labels'])
import logddd
for key in batch.keys():
    logddd.log(batch[key])