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
print("total GPU memory: %f GB" % (torch.cuda.mem_get_info()[1] / 1024 ** 3))
print("available GPU memory %f GB" % (torch.cuda.mem_get_info()[0] / 1024 ** 3))
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logging.warning("No GPU available, using CPU will be super slow")


# ### Load dataset, model, and tokenizer

# In[3]:


dataset_id = "ag_news"
model_id = "roberta-base"


# In[4]:


dataset = load_dataset(dataset_id)


# In[5]:


train_dataset = dataset['train']
test_dataset = dataset["test"].shard(num_shards=2, index=0)
val_dataset = dataset["test"].shard(num_shards=2, index=1)


# In[6]:


tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256, return_tensors="pt")

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# In[7]:


num_labels = dataset['train'].features['label'].num_classes
class_names = dataset["train"].features["label"].names

print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")


# In[8]:


id2label = {i: label for i, label in enumerate(class_names)}
config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})
config


# In[10]:


roberta_model = RobertaForSequenceClassification.from_pretrained(model_id, config=config).to(device)


# ### Evaluate performance before fine-tuning

# In[9]:


roberta_predictions = []
for i in tqdm(range(len(test_dataset["text"])), total=len(test_dataset["text"])):
    test_input = tokenizer(test_dataset["text"][i], return_tensors="pt").to(device)
    with torch.no_grad():
        logits = roberta_model(**test_input).logits
    predicted_class_id = logits.argmax().item()
    roberta_predictions.append(predicted_class_id)


# In[10]:


print("raw roberta accuracy: ", round(accuracy_score(test_dataset["label"], roberta_predictions), 3))


# ### Fine-tune Roberta

# In[11]:


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


# In[12]:


trainer = Trainer(
    model=roberta_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


# In[15]:


with torch.autocast(device, cache_enabled=False):
    trainer.train()


# In[16]:


trainer.evaluate()


# In[17]:


finetuned_roberta_outputs = trainer.predict(test_dataset)


# In[18]:


finetuned_roberta_predictions = finetuned_roberta_outputs[1]


# In[19]:


print("fine-tuned roberta accuracy: ", round(accuracy_score(test_dataset["label"], finetuned_roberta_predictions), 3))


# ### Load fine-tuned Roberta

# In[20]:


finetuned_model = RobertaForSequenceClassification.from_pretrained("./result/fine_tuned_agnews/checkpoint-667/", local_files_only=True)

