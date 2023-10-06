#!/usr/bin/env python
# coding: utf-8
import os
import torch
from tqdm import tqdm 
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import logging
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification, # added peft layers and hard linked to the project folder 
    TrainingArguments,
    Trainer,
    AutoModel,
    AutoConfig,
)
import json
import loralib
from utils import *

print("total GPU memory: %.2f GB" % (torch.cuda.mem_get_info()[1] / 1024 ** 3))
print("available GPU memory %.2f GB" % (torch.cuda.mem_get_info()[0] / 1024 ** 3))
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logging.warning("No GPU available, using CPU will be super slow")


# ### Load dataset, model, and tokenizer
dataset_id = "ag_news"
model_id = "roberta-base"
save_dir = "./results/lora_roberta_agnews"
os.makedirs(save_dir, exist_ok=True)
peft_config = {"lora": True,
               "lora_alpha": 2,
               "lora_dropout": 0.1,
               "lora_r": 2,
               "lora_bias": True,
               "layers_to_replace": ["query, key"],
               "monarch": False,
            #    "rank": None,
            #    "nblocks": None
               }

dataset = load_dataset(dataset_id)
train_dataset = dataset['train']
test_dataset = dataset["test"].shard(num_shards=2, index=0)
val_dataset = dataset["test"].shard(num_shards=2, index=1)

tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256, return_tensors="pt")

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

num_labels = dataset['train'].features['label'].num_classes
class_names = dataset["train"].features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# update dataset class names 
id2label = {i: label for i, label in enumerate(class_names)}
config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})
config.update(peft_config)
json.dump(peft_config, open(save_dir + "/peft_config.json", "w"))

# load model and init lora
roberta_model = RobertaForSequenceClassification.from_pretrained(model_id, config=config).to(device)
loralib.mark_only_lora_as_trainable(roberta_model)
last_name = None
param_stats(roberta_model, training=True)

# ### Evaluate performance before fine-tuning
with torch.autocast(device):
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
    output_dir=save_dir,
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


trainer.evaluate() # calls model.eval(), which causes lora layers to merge 
finetuned_roberta_outputs = trainer.predict(test_dataset)
finetuned_roberta_predictions = finetuned_roberta_outputs[1]
print("fine-tuned roberta accuracy: ", round(accuracy_score(test_dataset["label"], finetuned_roberta_predictions), 3))
torch.save(loralib.lora_state_dict(roberta_model), os.path.join(save_dir, "model.pt"))

# # ### Load fine-tuned Roberta
# finetuned_model = RobertaForSequenceClassification.from_pretrained(model_id, config=config).to(device)
# lora_weights = save_dir
# finetuned_model = PeftModel.from_pretrained(
#     finetuned_model, 
#     lora_weights, 
#     device_map="auto",
#     offload_folder="offload", 
# )
