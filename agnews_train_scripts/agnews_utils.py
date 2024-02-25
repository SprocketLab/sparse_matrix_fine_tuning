from transformers import (
    RobertaTokenizerFast,
    TrainingArguments,
    AutoModel,
    AutoConfig,
    RobertaForSequenceClassification,
    Trainer
) 
import loralib
import torch
from datasets import load_dataset
import json

def param_stats(model, training=False, print_trainable=False):
    param_count = 0
    param_trainable = 0
    model_size = 0
    
    for name, param in model.named_parameters():
        param_count += torch.numel(param) 
        model_size += torch.numel(param) * param.element_size()
        if param.requires_grad:
            param_trainable += torch.numel(param) 
            if print_trainable:
                print("trainable:", name)            
                
    # print("Total GPU memory: %.2f GB" % (torch.cuda.mem_get_info()[1] / 1024 ** 3))
    # print("Avail GPU memory %.2f GB" % (torch.cuda.mem_get_info()[0] / 1024 ** 3))
    print(
        f"Total parameters: {param_count / 1024 ** 2:.2f}M,\n \
        trainable parameters: {param_trainable / 1024 ** 2:.2f}M ({100 * param_trainable / param_count:.2f}%)\n \
        model size: {model_size / 1024 ** 2:.2f}MB"
    )
    if training:
        assert param_trainable != 0, "There's a bug in your code, your're training nothing!"


def prep_data(dataset_id, tokenizer):
    """
    Load dataset from huggingface and map to tensor ids
    """
    if dataset_id == "ag_news":
        dataset = load_dataset(dataset_id)
        input_key = "text"
    elif dataset_id in ["cola"]: # other datasets are complicated; use run_glue.py 
        dataset = load_dataset('glue', dataset_id)
        input_key = "sentence"
    else:
        raise ValueError("dataset not supported")
    
    train_dataset = dataset['train']
    test_dataset = dataset["test"].shard(num_shards=2, index=0)
    val_dataset = dataset["test"].shard(num_shards=2, index=1)

    tokenize = lambda batch : tokenizer(batch[input_key], padding=True, truncation=True, max_length=256, return_tensors="pt")
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    return dataset, train_dataset, val_dataset, test_dataset
    

def setup_trainer(model_id, dataset_id, save_dir, train_config, peft_config={}, device="cuda"):
    """
    Setup trainer for finetuning on ag_news dataset
    Args:
        train_config: training hyperparams
        peft_config: model configs setting PEFT (lora, monarch)
    """
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
    json.dump(peft_config, open(save_dir + "/peft_config.json", "w")) # save peft config for record

    # load model and init peft layers
    roberta_model = RobertaForSequenceClassification.from_pretrained(model_id, config=config).to(device)
    if peft_config['monarch']:
        roberta_model.roberta.set_peft_config(peft_config) 
    elif peft_config['lora']:
        loralib.mark_only_lora_as_trainable(roberta_model)

    # Set up hyperparams for finetuning
    train_config["output_dir"] = save_dir
    training_args = TrainingArguments(
        **train_config
    )

    trainer = Trainer(
        model=roberta_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    return roberta_model, trainer, test_dataset, config


def run_trainer(trainer, test_dataset, precision="fp16", device="cuda"):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    assert precision in dtype.keys(), "precision must be one of fp16, bf16, fp32"
    dtype = dtype[precision]
    
    with torch.autocast(device, cache_enabled=False, dtype=dtype):
        trainer.train() # can disable some dense layers 
        
        trainer.evaluate()
        
        # test set eval
        t1 = time.time()
        finetuned_roberta_outputs = trainer.predict(test_dataset)
        print("Inferece time: ", time.time() - t1)
        
        finetuned_roberta_predictions = finetuned_roberta_outputs[1]
        print("fine-tuned roberta accuracy: ", round(accuracy_score(test_dataset["label"], finetuned_roberta_predictions), 3))

