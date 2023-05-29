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
# https://zhuanlan.zhihu.com/p/607988185
def tokenize_function(examples,tokenizer):
    """code/
    将data转为token
    :param examples:data
    """
    label = []
    data = []
    res = []
    for example in examples:
        result = tokenizer(example.split('→')[0],return_tensors="pt",padding=True, truncation=True,)
        labels = tokenizer.convert_tokens_to_ids(example.split("→")[1])
        result["labels"] = [labels]
        # if tokenizer.is_fast:
        #     result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        res.append(result)

    return res


    
if __name__ == '__main__':
    datas = ['在句子"脉细弦"中，词语"脉"的前文如果是由"[CLS]"词性的词语"[CLS]"来修饰，那么词语"脉"的词性是"[MASK]"→ NR', '在句子"脉细弦"中，词语"细"的前文如果是由"NR"词性的词语"脉"来修饰，那么词语"细"的词性是"[MASK]"→ VA', '在句子"脉细弦"中，词语"弦"的前文如果是由"VA"词性的词语"细"来修饰，那么词语"弦"的词性是"[MASK]"→ VA']


    device = torch.device('cpu')
    model_checkpoint =  "/home/dlf/prompt/code/model/bert_large_chinese"
    # model_checkpoint = "distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    batch_size = 32
    train,valid,test = get_all_data()
    # train = datas
    # ======================
    tokenized_train = tokenize_function(train,tokenizer)
    # print(tokenized_train)
    # exit(0)
    # for item in tokenized_train:
    #     print(item)
        
    #     exit(0)
    # ========================
    # Show the training loss with every epoch
    logging_steps = len(train) // batch_size
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
        fp16=False,
        logging_steps=logging_steps,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.0)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenize_function(train,tokenizer),
        eval_dataset=tokenize_function(valid,tokenizer),
        data_collator=data_collator,
    )

  
    train_dataloader = DataLoader(
        tokenize_function(train,tokenizer),
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    eval_dataloader = DataLoader(
        tokenize_function(valid,tokenizer),
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    # print(train_dataloader)
    # for batch in train_dataloader:
    #     print(batch)
 
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # accelerator = Accelerator()
    # model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader
    # )

    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # model = model.to(device)
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            # batch = torch.tensor(batch, dtype=torch.long, device=device)
            data = batch
            print(data)
            exit(0)
            outputs = model(**data)
            loss = outputs.loss
            # accelerator.backward(loss)
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
                data = batch
                outputs = model(**data)

            loss = outputs.loss
            # losses.append(accelerator.gather(loss.repeat(batch_size)))
            losses.append(loss)

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

        # Save and upload
        # accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        # if accelerator.is_main_process:
        #     tokenizer.save_pretrained(output_dir)
        #     repo.push_to_hub(
        #         commit_message=f"Training in progress epoch {epoch}", blocking=False
        #     )