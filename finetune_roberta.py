#!/usr/bin/env python
# coding: utf-8


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm 
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import logging
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModel,
    AutoConfig,
)
from train_utils import * 
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logging.warning("No GPU available, using CPU will be super slow")


# ### Load dataset, model, and tokenizer
dataset_id = "ag_news"
model_id = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

# split dataset 
dataset, train_dataset, val_dataset, test_dataset = prep_data(dataset_id, tokenizer)
num_labels = dataset['train'].features['label'].num_classes
class_names = dataset["train"].features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# update training classes 
id2label = {i: label for i, label in enumerate(class_names)}
config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})

roberta_model = RobertaForSequenceClassification.from_pretrained(model_id, config=config).to(device)
param_stats(roberta_model, training=True)

# ### Evaluate performance before fine-tuning
roberta_predictions = []
for i in tqdm(range(len(test_dataset["text"])), total=len(test_dataset["text"])):
    test_input = tokenizer(test_dataset["text"][i], return_tensors="pt").to(device)
    with torch.no_grad():
        logits = roberta_model(**test_input).logits
    predicted_class_id = logits.argmax().item()
    roberta_predictions.append(predicted_class_id)
print("raw roberta accuracy: ", round(accuracy_score(test_dataset["label"], roberta_predictions), 3))


# ### Fine-tune Roberta
training_args = TrainingArguments(
    output_dir="./results/fine_tuned_agnews",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=180,
    per_device_eval_batch_size=380,
    learning_rate=0.005,
    weight_decay=0.01,
    logging_strategy="steps",
    warmup_steps=50,
    eval_steps=100,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=roberta_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
with torch.autocast(device, cache_enabled=False):
    trainer.train()

    # test set eval
    trainer.evaluate()
    t1 = time.time()
    finetuned_roberta_outputs = trainer.predict(test_dataset)
    print("Inferece time: ", time.time() - t1)
    finetuned_roberta_predictions = finetuned_roberta_outputs[1]
    print("fine-tuned roberta accuracy: ", round(accuracy_score(test_dataset["label"], finetuned_roberta_predictions), 3))

# Check loading Load fine-tuned Roberta
finetuned_model = RobertaForSequenceClassification.from_pretrained("./results/fine_tuned_agnews/checkpoint-667/", local_files_only=True)

