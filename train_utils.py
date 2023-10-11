import torch
from datasets import load_dataset
from src.models.layers.monarch_linear import MonarchLinear

def param_stats(model, training=False, print_trainable=False):
    param_count = 0
    param_trainable = 0
    model_size = 0
    
    for name, param in model.named_parameters():
        param_count += torch.numel(param) 
        model_size += torch.numel(param) * param.element_size()
        if param.requires_grad:
            param_trainable += torch.numel(param) 
            print("trainable:", name)            
            
    print("Total GPU memory: %.2f GB" % (torch.cuda.mem_get_info()[1] / 1024 ** 3))
    print("Avail GPU memory %.2f GB" % (torch.cuda.mem_get_info()[0] / 1024 ** 3))
    print(
        f"Total parameters: {param_count / 1024 ** 2:.2f}M,\n \
        trainable parameters: {param_trainable / 1024 ** 2:.2f}M ({100 * param_trainable / param_count:.2f}%)\n \
        model size: {model_size / 1024 ** 2:.2f}MB"
    )
    if training:
        assert param_trainable != 0, "There's a bug in your code, your're training nothing!"


def select_gpu():
    """
    Select the gpu with maximum free memory
    """
    num_gpus = torch.cuda.device_count()
    max_mem = 0
    max_gpu = 0
    for device in range(num_gpus):
        torch.cuda.set_device(device)
        free_mem = torch.cuda.mem_get_info()[0]
        if free_mem > max_mem:
            max_mem = free_mem  
            max_gpu = device
            
    torch.cuda.set_device(max_gpu)
    print("Selected GPU: %d" % max_gpu)
    
def prep_data(dataset_id, tokenizer):
    
    dataset = load_dataset(dataset_id)
    train_dataset = dataset['train']
    test_dataset = dataset["test"].shard(num_shards=2, index=0)
    val_dataset = dataset["test"].shard(num_shards=2, index=1)

    tokenize = lambda batch : tokenizer(batch["text"], padding=True, truncation=True, max_length=256, return_tensors="pt")
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    return dataset, train_dataset, val_dataset, test_dataset
    
