import argparse
import gc
import os
import sys
import warnings

import torch
from transformers import Seq2SeqTrainer, Trainer, TrainerCallback
from transformers.utils.import_utils import is_sagemaker_mp_enabled

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # add current directory to path
import glob
import json
import logging
import math
import shutil
import time
from ast import literal_eval
from collections import defaultdict
from functools import partial
from os.path import exists, isdir, join
from typing import Dict, List, Union

import bitsandbytes as bnb
import torch.nn as nn
from ray import tune

import wandb
from peft import BOFTConfig, LoraConfig, get_peft_model
from src.models.layers.monarch_linear import MonarchLinear, Scaler

PEFT_ROBERTA_PATH = "/fly/task_configs/monarch_roberta_glue/peft_config.json"
PEFT_DEBERTA_PATH = "/fly/task_configs/glue_deberta/peft_monarch_deberta.json"
PEFT_ROBERTA_LORA_PATH = "/fly/task_configs/lora_roberta_glue/peft_config.json"
# PEFT_DEBERTA_PATH = "/workspace/private/sparse_matrix_fine_tuning/task_configs/glue_deberta/peft_monarch_deberta.json"
PEFT_DEBERTA_BOFT_PATH = "./task_configs/glue_deberta/peft_boft_deberta.json"
PEFT_ROBERTA_BOFT_PATH = "./task_configs/monarch_roberta_glue/peft_boft_roberta.json"


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run GLUE with additional arguments",
    )

    # Add the positional argument for the config path
    parser.add_argument(
        "config_path", help="path to the GLUE task config file under task_configs/monarch_roberta_glue/peft_con"
    )

    # Add optional arguments
    parser.add_argument(
        "--use_monarch", default=True, type=eval, help="Use monarch. Mostly you want this (default: True)"
    )
    parser.add_argument("--use_boft", default=False, type=bool, help="Use BOFT")
    parser.add_argument("--use_lora", default=False, type=eval)
    parser.add_argument(
        "--do_tune", default=False, type=eval, help="Whether to do Hyperparameter optimization (HPO) using ray tune."
    )
    parser.add_argument("--wandb", default=True, type=eval, help="Use Weights & Biases for logging")
    parser.add_argument(
        "--adapter", default=True, type=eval, help="Use lora adapter style. If false will project dense to sparse "
    )
    parser.add_argument("--tune_unit", default="eval_iter", help="Budget unit for HPO.", choices=["time", "eval_iter"])
    parser.add_argument("--n_trials", default=25, type=int, help="Number of trials for HPO")
    parser.add_argument("--gpus_per_trial", default=0.5, type=float, help="Number of GPUs to use per HPO trial")
    parser.add_argument("--tune_blk_config", default=False, type=eval, help="Whether to tune block sizes & rank ")

    # Wandb & Ray Tune
    parser.add_argument("--group", default="", help="For grouping wandb runs")
    parser.add_argument("--notes", default="", help="Notes to add to wandb run name. This won't mess up best HP group")
    parser.add_argument("--project", default=None, help="For grouping wandb groups and runs")
    parser.add_argument("--full_group", default=None, help="Full group name for resuming eval (with date and task)")
    parser.add_argument(
        "--time", default=None, help="For grouping wandb groups and runs. If not provided will use current time"
    )
    parser.add_argument(
        "--as_base_hp",
        default=False,
        type=eval,
        help="For HP tuning only. \
                                Whether to save an extra copy in the dataset folder, which will be used by other un-tuned runs default",
    )
    parser.add_argument("--resume", default=False, type=eval, help="Whether to resume Ray Tune from error")
    parser.add_argument(
        "--load_group",
        default=False,
        type=eval,
        help="Whether to load the full group name from group dir's full_group.txt",
    )
    parser.add_argument("--profile", action="store_true", help="Whether to profile performance")
    parser.add_argument("--disable_tqdm", default=False, type=eval, help="Disable trainer progress bar")
    args, unknown = parser.parse_known_args()
    return args


def load_best_hp(run_dir, task_dir="nonexistent"):
    best_hyperparams = None
    best_hp_path = os.path.join(run_dir, "best_hyperparams.json")
    base_hp_path = os.path.join(task_dir, "best_hyperparams.json")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    if os.path.exists(best_hp_path):
        best_hyperparams = json.load(open(best_hp_path))
        print(f"Using best hp: {best_hyperparams}")
    elif os.path.exists(base_hp_path):
        best_hyperparams = json.load(open(base_hp_path))
        print(f"Using best hp for from the base task dir: {best_hyperparams}")
    else:
        print("No best hyperparameters found.")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    return best_hyperparams


def print_dtypes(model):
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)


def param_stats(model, training=True, print_trainable=False, skip_cls=True):
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
        f"Total parameters: {param_count / 1024 ** 2:.3f}M,\n \
        trainable parameters: {param_trainable / 1024 ** 2:.3f}M ({100 * param_trainable / param_count:.3f}%)"
    )

    if training:
        assert param_trainable != 0, "There's a bug in your code, your're training nothing!"
    return param_trainable


def replace_with_symlink(path: str, target_disk: str):
    """
    For replacing all model checkpoints in a dir with symlinks, and moving them to a new disk.
    """
    if not os.path.exists(target_disk):
        print(f"{target_disk} does not exist, skipping replacement")
        return
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
    if new_args is None or len(new_args) == 0:
        return
    extra_args = {}
    new_args = new_args.items() if isinstance(new_args, dict) else new_args

    for arg in new_args:

        if isinstance(arg, tuple):
            # dictionary
            key, val = arg
        elif arg.startswith("--"):
            # command line args
            try:
                key, val = arg.split("=") if "=" in arg else arg.split(" ")
            except Exception as e:
                print(f"Error: {e} for command line arg {arg}")
            key = key[2:]
        else:
            raise ValueError(f"wrong format for {arg}, extra command line argument must be --key=value or --key value")

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
                if not (isinstance(attempt, type(config[key])) or config[key] is None):
                    warnings.warn(f"wrong type for {key}, expected {type(config[key])}, got {type(attempt)}")

                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                config[key] = attempt
                exists = True

        # Use this as global variable?
        if not exists:
            extra_args[key] = attempt
    return extra_args


def get_run_group(
    task_name: str = None, do_tune: bool = False, group: str = None, cur_time: str = None, notes: str = None
):
    """
    Get wandb run group. If time is provided, will keep those tasks in the same time group in wandb.
    """
    run_group = "tune" + "_" if do_tune else ""  # if hyperapram tuning, add tune to group name
    run_group += task_name + "_" if task_name else ""
    if notes:
        run_group += notes + "_"
    run_group += group + "_" if group else ""
    run_group += time.strftime("%m-%d-%H", time.localtime()) if cur_time is None else cur_time
    return run_group


class MySeq2SeqTrainer(Seq2SeqTrainer):
    def save_model(self, output_dir=None, _internal_call=False):
        """Only save trainable params"""
        if output_dir is None:
            output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        state_dict = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                state_dict[n] = p.data
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


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
        # if hasattr(self.model, "roberta") and self.train_dataset is not None:
        if (
            hasattr(self.model, "roberta") or hasattr(self.model, "deberta")
        ) and self.train_dataset is not None:  # EDIT
            # if self.train_dataset is not None:
            len_dataloader = len(self.get_train_dataloader()) // self.args.gradient_accumulation_steps
            self.num_training_steps = math.ceil(len_dataloader * self.args.num_train_epochs)

    def training_step(self, model, inputs):
        [param.shape for param_group in self.optimizer.param_groups for param in param_group["params"]]
        # if not any([len(shape) == 3 for shape in param_shapes]):
        #     self.create_optimizer_and_scheduler(self.num_training_steps)
        #     print("Recreating optimizer for monarch params")
        # Check param count
        if self.train_step % self.log_param_steps == 0:
            param_stats(self.model, training=True, print_trainable=False, skip_cls=True)
        self.train_step += 1
        return super().training_step(model, inputs)

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        """Only save trainable params"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        state_dict = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                state_dict[n] = p.data
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    def create_optimizer(self):
        """
        Modified from https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/trainer.py#L923
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        # if self.optimizer is None:
        no_decay = ["bias", "LayerNorm.weight"]
        large_lr = (
            [
                "scaler",
            ]
            if self.use_scaler
            else ["blkdiag2", "blkdiag_mult"]
        )

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
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay + large_lr) and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in large_lr) and p.requires_grad
                ],
                "lr": new_lr,
                "weight_decay": new_decay,
            },
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

        return self.optimizer


########################## PEFT module replacement helpers ##########################
def init_boft(
    model,
    peft_config,
):
    boft_config = BOFTConfig(
        boft_block_size=peft_config["boft_block_size"],  # These are mutually exclusive
        boft_block_num=peft_config["boft_block_num"],
        boft_n_butterfly_factor=peft_config["boft_n_butterfly_factor"],
        target_modules=peft_config["layers_to_adapt"],
        boft_dropout=peft_config["boft_dropout"],
        bias=peft_config["bias"],
    )
    model = get_peft_model(model, boft_config)
    model = model.base_model.model  # remove the wrappers. NOTE without this, trainer won't return eval metrics, damn..
    # model = model.base_model.model.to(peft_config["dtype"])
    # Unfreeze the classification head; pooler and classifier
    for n, p in model.named_parameters():
        n_split = n.split(".")
        if (
            ("pooler" in n_split) or ("classifier" in n_split) or ("lm_head" in n_split)
        ):  # will be deberta.pooler / deberta.classifier. lm_head for language models
            p.requires_grad = True
            print("Unfroze ", n)
    return model


def init_lora(model, peft_config):
    lora_config = LoraConfig(**peft_config)
    model = get_peft_model(model, lora_config)
    return model


adapted_layers = set()


class peft_module:
    """A helper module to automatically replace layers with Monarch matrices"""

    def set_monarch_recursive(self):
        if not self.peft_config.get("monarch", False):
            return
        global adapted_layers

        def factor(n):
            # find factors closest to sqrt(n)
            sizes = [(i, n // i) for i in range(1, math.floor(math.sqrt(n)) + 1) if n % i == 0][-1]
            # Larger factor first (nblocks) -> memory bound. Smaller factor first -> compute bound
            sizes = (sizes[0], sizes[1])
            return sizes

        for module in self.modules():
            for name in self.peft_config["layers_to_adapt"]:
                layer = getattr(module, name, None)
                if layer is None or isinstance(layer, MonarchLinear):
                    continue

                weights = layer.weight
                m, n = weights.shape

                if self.peft_config["adapter"] and self.nblocks != "sqrt(n)":
                    # freeze dense, init and train monarch, and then merge during inference
                    nblocks = self.nblocks
                else:
                    # project dense to monarch and keep monarch only
                    nblocks = factor(layer.in_features)[0]  # increase to sqrt(n) blocks -> more params
                    if self.nblocks == "sqrt(n)":
                        self.nblocks = nblocks

                bias = layer.bias != None
                new_layer = MonarchLinear(
                    in_features=n,
                    out_features=m,
                    nblocks=nblocks,
                    weights=weights,
                    bias=bias,
                    peft_config=self.peft_config,
                )
                if bias:
                    new_layer.bias = layer.bias

                # Disable dense grads
                if hasattr(new_layer, "dense"):
                    new_layer.dense.requires_grad_(False)
                if new_layer.bias is not None:
                    new_layer.bias.requires_grad_(False)
                setattr(module, name, new_layer)

                # For printing info
                adapted_layers.add((name, (m, n), new_layer.blkdiag1.shape, new_layer.blkdiag2.shape))

    def set_peft_config(self, peft_config):
        self.peft_config = peft_config
        if peft_config["monarch"]:
            self.rank = peft_config["blk_r"]
            self.nblocks = peft_config["nblocks"]


def init_monarch_layers(model: nn.Module, peft_config: Dict, target_classes: Union[List[str], List[nn.Module]] = []):
    """
    Hack the model by replacing modules with Monarch adapters
    Args:
        target_classes: List of classes look recursively into for peft_config["layers_to_adapt"].
            If not givenss, will traverse all layers.
    """
    # TODO: return trainable params for optimizer
    if getattr(model, "monarch_param_set", False) and peft_config == getattr(model, "peft_config", None):
        print("Monarch layers already initialized")
        return

    setattr(model, "set_peft_config", partial(peft_module.set_peft_config, model))
    model.set_peft_config(peft_config)

    for name, module in model.named_modules():
        # Replace linear with monarch if is target layer
        if any(hasattr(module, _name) for _name in peft_config["layers_to_adapt"]):
            # hack the module to "inherit" peft_adapter
            setattr(module, "set_monarch_recursive", partial(peft_module.set_monarch_recursive, module))
            setattr(module, "set_peft_config", partial(peft_module.set_peft_config, module))
            module.set_peft_config(peft_config)
            module.set_monarch_recursive()

        # Only enable grads for adapters
        if isinstance(module, MonarchLinear) or isinstance(module, Scaler) or "classifier" in name:
            module.requires_grad_(True)
            if hasattr(module, "dense"):
                module.dense.requires_grad_(False)
        else:
            module.requires_grad_(False)
    model.monarch_param_set = True

    global adapted_layers
    for name, old_shape, shape_1, shape_2 in adapted_layers:
        print(f"Adapted {name} {old_shape} with monarch layers: {shape_1}, {shape_2}")


def get_hpo_metric(target_metric: str, metrics: dict):
    return metrics[target_metric]


def watch_layers(model, max_per_module=2):
    """Monitor how weights are updated"""
    if wandb.run != None or tune.is_session_enabled():
        return
    print("Enabling wandb watch")
    watch_count = defaultdict(int)
    # log modules of interest
    for name, module in model.named_modules():
        if isinstance(module, MonarchLinear) or isinstance(module, Scaler):
            layer_name = name.split(".")[-1]
            if watch_count[(type(module), layer_name)] < max_per_module:
                try:
                    wandb.watch(module, log="parameters", log_freq=300)
                except ValueError:
                    # Sometimes throws weird wandb uninitialized bug when used with Ray Tune...
                    return
                watch_count[(type(module), layer_name)] += 1

    for (module, layer_name), count in watch_count.items():
        print(f"Watched {count} {layer_name} layers  ")

    # Log all py files for reference
    files_to_log = ["qlora_monarch.py", "monarch_linear.py", "train_utils.py"]

    for file in glob.glob("/fly/**/*.py", recursive=True):
        if any([file.endswith(f) for f in files_to_log]):
            artifact = wandb.Artifact(os.path.basename(file), type="code")
            artifact.add_file(file)
            wandb.run.log_artifact(artifact)


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(abbreviated=True)) # Also see https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/7


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
        # peak memory
        # print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")


# Example:
# with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
#                                         torch.profiler.ProfilerActivity.CUDA],
#                             schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2, repeat=2),
#                             on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-training-trainer'),
#                             profile_memory=True,
#                             with_stack=True,
#                             record_shapes=True) as prof:

#     trainer.add_callback(ProfCallback(prof=prof))
#     trainer.train()

# print(f'training time, {(time.perf_counter() - start):.1f} s')


def set_merged(model):
    for name, module in model.named_modules():
        if isinstance(module, MonarchLinear):
            module.merged = True


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith("checkpoint"):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def find_all_linear_names(model, nbits=16):
    cls = bnb.nn.Linear4bit if nbits == 4 else (bnb.nn.Linear8bitLt if nbits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
