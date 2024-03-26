import torch
from transformers import Trainer
import argparse
from typing import Optional
from transformers.utils.import_utils import is_sagemaker_mp_enabled
import warnings 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))) # add current directory to path
import shutil
import math
import time
import logging
import torch.nn as nn
from tqdm import tqdm
from ast import literal_eval
from typing import Dict, List, Union



def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Run GLUE with additional arguments", )

    # Add the positional argument for the config path
    parser.add_argument("config_path", help="path to the GLUE task config file under task_configs/glue_peft_configs")

    # Add optional arguments
    parser.add_argument("--use_monarch", default=True, type=eval, help="Use monarch. Mostly you want this (default: True)")
    parser.add_argument("--do_tune", default=False, type=eval, help="Whether to do Hyperparameter optimization (HPO) using ray tune.")
    parser.add_argument("--use_wandb", default=True, type=eval, help="Use Weights & Biases for logging")
    parser.add_argument("--adapter", default=True, type=eval, help="Use lora adapter style. If false will project dense to sparse ")
    parser.add_argument("--tune_unit", default="eval_iter", help="Budget unit for HPO.", choices=["time", "eval_iter"])
    parser.add_argument("--n_trials", default=36, type=int, help="Number of trials for HPO")
    parser.add_argument("--gpus_per_trial", default=0.5, type=float, help="Number of GPUs to use per HPO trial")
    parser.add_argument("--tune_blk_config", default=False, type=eval, help="Whether to tune block sizes & rank ")
    # Wandb grouping args
    parser.add_argument("--group", default="", help="For grouping wandb runs")
    parser.add_argument("--notes", default="", help="Notes to add to wandb run name. This won't mess up best HP group" )
    parser.add_argument("--project", default=None, help="For grouping wandb groups and runs")
    parser.add_argument("--full_group", default=None, help="Full group name for resuming eval (with date and task)")
    parser.add_argument("--time", default=None, help="For grouping wandb groups and runs. If not provided will use current time")
    parser.add_argument("--as_base_hp", default=False, type=eval, help="For HP tuning only. \
                                Whether to save an extra copy in the dataset folder, which will be used by other un-tuned runs default")
    parser.add_argument("--resume_tune", default=False, type=eval, help="Whether to resume Ray Tune from error")
    parser.add_argument("--load_group", default=False, type=eval, help="Whether to load the full group name from group dir's full_group.txt")
    parser.add_argument("--move_ckpt", default=False, type=eval, help="Replace the final checkpoints with symlinks to another disk to save space")
    args, unknown = parser.parse_known_args()
    return args


def param_stats(model, training=True, print_trainable=False, skip_cls=False):
    """
    Do a param count and optionally print the trainable layers
    """
    param_count = 0
    param_trainable = 0
    model_size = 0
    
    for name, param in model.named_parameters():
        param_count += torch.numel(param) 
        model_size += torch.numel(param) * param.element_size()
        
        if param.requires_grad:
            if "classifier" in name:
                if skip_cls:
                    continue 
            
            param_trainable += torch.numel(param) 
            if print_trainable:
                print(name, f": {torch.numel(param) / 1024 ** 2:.4f}M, {param.shape}")            
                
            
    # print("Total GPU memory: %.2f GB" % (torch.cuda.mem_get_info()[1] / 1024 ** 3))
    # print("Avail GPU memory %.2f GB" % (torch.cuda.mem_get_info()[0] / 1024 ** 3))
    print(
        f"Total parameters: {param_count / 1024 ** 2:.2f}M,\n \
        trainable parameters: {param_trainable / 1024 ** 2:.2f}M ({100 * param_trainable / param_count:.2f}%)\n \
        model size: {model_size / 1024 ** 2:.2f}MB"
    )
    
    if training:
        assert param_trainable != 0, "There's a bug in your code, your're training nothing!"



def replace_with_symlink(path: str, target_disk: str):
    """
    For replacing all model checkpoints in a dir with symlinks, and moving them to a new disk.
    """
    if not os.path.exists(target_disk):
        print(f"{target_disk} does not exist, skipping replacement")
        return 
    breakpoint()
    # replace the first two levels
    cur_disk = path.split("/")[0]
    if cur_disk == target_disk:
        return 
    new_path = path.replace(cur_disk, target_disk)

    # check if they link to the same file or there's a symlink already
    if os.path.exists(new_path) and os.path.samefile(path, new_path):
        print(f"{path} and {new_path} are the same file, skipping replacement")
        return
    
    # move to new disk to clear space
    new_dir = os.path.dirname(new_path)
    os.makedirs(new_dir, exist_ok=True) 
    shutil.move(path, new_path)
    os.symlidnk(new_path, path)
    
    

def override_config(old_configs: List[Dict], new_args: Union[List[str], Dict]):
    """
    Scan through the old configs and update them with new args if they exist.
    """
    extra_args = {}
    new_args = new_args.items() if isinstance(new_args, dict) else new_args
    
    for arg in new_args:
        
        if isinstance(arg, tuple):
            # dictionary
            key, val = arg
        elif arg.startswith('--'):
            # command line args
            key, val = arg.split('=')
            key = key[2:]
        else:
            raise ValueError(f"wrong format for {arg}, extra command line argument must be --key=value")

        try:
            # attempt to eval it it (e.g. if bool, number, or etc)
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            # if that goes wrong, just use the string
            attempt = val
            
        exists = False
        for config in old_configs:
            if type(config) is not dict:
                # Trying fetching dict from HF config                
                config = config.__dict__
                
            if key in config.keys():
                if not(isinstance(attempt, type(config[key])) or config[key] is None):
                    warnings.warn(f"wrong type for {key}, expected {type(config[key])}, got {type(attempt)}")

                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                config[key] = attempt
                exists = True
        
        # Use this as global variable?
        if not exists:
            extra_args[key] = attempt
    return extra_args
        
        
def get_run_group(task_name: str=None, do_tune: bool=False, group: str=None, cur_time: str=None, notes: str=None):
    """
    Get wandb run group. If time is provided, will keep those tasks in the same time group in wandb.
    """
    run_group = "tune" + "_" if do_tune else "" # if hyperapram tuning, add tune to group name
    run_group += task_name + "_" if task_name else ""
    if notes:
        run_group += notes + "_"
    run_group += group + "_" if group else ""
    run_group += time.strftime("%m-%d-%H", time.localtime()) if cur_time is None else cur_time 
    return run_group


class MyAwesomeTrainer(Trainer):
    """
    Modified for initializing the monarch params and adding them to the optimizer
    before the 1st training step.
    """
    
    def __init__(self, *args, **kwargs):
        self.large_lr = kwargs.pop("large_lr", False)
        self.use_scaler = kwargs.pop("use_scaler", False)
        self.new_lr = kwargs.pop("new_lr", 5e-3)
        self.log_param_steps = kwargs.pop("log_param_steps", 900)
        self.train_step = 0
        
        super().__init__(*args, **kwargs)
        if hasattr(self.model, "roberta") and self.train_dataset is not None:
            # A bit ugly...but is for re-initializing optimizer after adding monarch params
            self.model.roberta.trainer = self 
            len_dataloader = len(self.get_train_dataloader()) // self.args.gradient_accumulation_steps
            self.num_training_steps = math.ceil(len_dataloader * self.args.num_train_epochs)
            

    def training_step(self, model, inputs):
        # Assignment is faster than if; just do it every time. Must assign this way in each Ray Tune run
        self.model.roberta.trainer = self 
        self.model.train() # From my tests, this line's speed is model size agnostic
        
        # Check param count
        if self.train_step % self.log_param_steps == 0:
            param_stats(self.model, training=True, print_trainable=False, skip_cls=True)
        self.train_step += 1
        return super().training_step(model, inputs)
    
    
    def create_optimizer(self):
        """
        Modified from https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/trainer.py#L923
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        # if self.optimizer is None:
        no_decay = ["bias", "LayerNorm.weight"]
        large_lr = ["scaler",] if self.use_scaler else ["blkdiag2", "blkdiag_mult"]
        
        if self.large_lr:
            new_lr = self.new_lr
            new_decay = 0 if self.use_scaler else self.args.weight_decay
            print(f"Using lr {new_lr} and weight decay {new_decay} for {large_lr}")
        else:
            new_lr = self.args.learning_rate 
            new_decay = self.args.weight_decay
            print("Using the same lr for all layers")
            
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay + large_lr)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in large_lr)],
                "lr": new_lr,
                "weight_decay": new_decay
            }
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    logging.info(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logging.debug(f"bitsandbytes: will optimize {module} in fp32")
            logging.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        return self.optimizer
    
