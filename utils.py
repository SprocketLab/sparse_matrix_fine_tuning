import torch

def param_stats(model, training=False):
    param_count = 0
    param_trainable = 0
    model_size = 0
    for name, param in model.named_parameters():
        param_count += torch.numel(param) 
        param_trainable += torch.numel(param) if param.requires_grad else 0
        model_size += torch.numel(param) * param.element_size()
            
    print(
        f"Total parameters: {param_count / 1024 ** 2:.2f} M,\n \
        trainable parameters: {param_trainable / 1024 ** 2:.2f} M ({100 * param_trainable / param_count:.2f}%)\n \
        model size: {model_size / 1024 ** 2:.2f} MB"
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