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
    RobertaForSequenceClassification, # added peft layers and hard linked to the project folder 
    TrainingArguments,
    Trainer,
    AutoModel,
    AutoConfig,
) 
import json
from train_utils import *
import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--rank", type=int, default=1)
# parser.add_argument("--nblocks", type=int, default=4)
# parser.add_argument("--lora", action="store_true")
# parser.add_argument("--monarch", action="store_true")
# parser.add_argument("--lora_alpha", type=float, default=2)


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logging.warning("No GPU available, using CPU will be super slow")


# ### Load dataset, model, and tokenizer
dataset_id = "ag_news"
model_id = "roberta-base"
save_dir = "./results/lora_roberta_agnews"
os.makedirs(save_dir, exist_ok=True)

# set training config
peft_config = {"lora": False, "monarch": True, "rank": 1, "nblocks": 4, "layers_to_replace": ["query", "value"]}
tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

dataset, train_dataset, val_dataset, test_dataset = prep_data(dataset_id, tokenizer)
num_labels = dataset['train'].features['label'].num_classes
class_names = dataset["train"].features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# update labels 
id2label = {i: label for i, label in enumerate(class_names)}
config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})
config.update(peft_config)
json.dump(peft_config, open(save_dir + "/peft_config.json", "w"))

# load model 
roberta_model = RobertaForSequenceClassification.from_pretrained(model_id, config=config).to(device)
roberta_model.init_monarch_layers() # project weights to monarch matrices
param_stats(roberta_model, training=True, print_trainable=True)


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


trainer.evaluate()
finetuned_roberta_outputs = trainer.predict(test_dataset)
finetuned_roberta_predictions = finetuned_roberta_outputs[1]
print("fine-tuned roberta accuracy: ", round(accuracy_score(test_dataset["label"], finetuned_roberta_predictions), 3))
torch.save(roberta_model.state_dict(), os.path.join(save_dir, "model.pt" ))

# ### Load fine-tuned Roberta
finetuned_model = RobertaForSequenceClassification.from_pretrained(model_id, config=config).to(device)
lora_weights = save_dir 
finetuned_model = PeftModel.from_pretrained(
    finetuned_model, 
    lora_weights, 
    device_map="auto",
    offload_folder="offload", 
)
