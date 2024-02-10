import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
    AutoModel,
    AutoConfig,
    get_scheduler
) 
import argparse
from typing import Optional
from transformers.utils.import_utils import is_sagemaker_mp_enabled
import warnings 
import loralib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))) # add current directory to path
from fly_src.models.modeling_roberta import RobertaForSequenceClassification
import math
import json
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from ast import literal_eval
from typing import Dict, List, Tuple
from ray import tune
from dataclasses import dataclass, field


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Run GLUE with additional arguments", )

    # Add the positional argument for the config path
    parser.add_argument("config_path", help="path to the GLUE task config file under task_configs/glue_peft_configs")

    # Add optional arguments
    parser.add_argument("--use_monarch", type=eval, default="True", help="Use monarch. Mostly you want this (default: True)")
    parser.add_argument("--do_tune", type=eval, default="False", help="Whether to do Hyperparameter optimization (HPO) using ray tune.")
    parser.add_argument("--use_wandb", type=eval, default="True", help="Use Weights & Biases for logging")
    parser.add_argument("--adapter", type=eval, default="True", help="Use lora adapter style. If false will project dense to sparse ")
    parser.add_argument("--tune_unit", default="eval_iter", help="Budget unit for HPO.", choices=["time", "eval_iter"])
    parser.add_argument("--group", default="", help="For grouping wandb runs")
    parser.add_argument("--project", default=None, help="For grouping wandb groups and runs")
    parser.add_argument("--full_group", default=None, help="Full group name for resuming eval (with date and task)")
    parser.add_argument("--time", default=None, help="For grouping wandb groups and runs. If not provided will use current time")
    parser.add_argument("--tune_decay", type=eval, default=True, help="Whether to tune weight decay in ASHA")
    args, unknown = parser.parse_known_args()
    return args


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


def select_gpu(exclude=[]):
    """
    Select the gpu with maximum free memory
    """
    num_gpus = torch.cuda.device_count()
    max_mem = 0
    max_gpu = 0
    for device in range(num_gpus):
        torch.cuda.set_device(device)
        free_mem = torch.cuda.mem_get_info()[0]
        if free_mem > max_mem and device not in exclude:
            max_mem = free_mem  
            max_gpu = device
            
    torch.cuda.set_device(max_gpu)
    print("Selected GPU: %d" % max_gpu, "with max memory %.2f GB" % (max_mem / 1024 ** 3))
    return max_gpu
    
    
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


def override_config(old_configs: List[Dict], new_args: List[str] or Dict):
    """Scan through the old configs and update them with new args if they exist
    """
    extra_args = {}
    new_args = new_args.items() if type(new_args) == dict else new_args
    
    for arg in new_args:
        
        if type(arg) == tuple:
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
                if not(type(attempt) == type(config[key]) or config[key] == None):
                    warnings.warn(f"wrong type for {key}, expected {type(config[key])}, got {type(attempt)}")

                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                config[key] = attempt
                exists = True
        
        # Use this as global variable?
        if not exists:
            extra_args[key] = attempt
    return extra_args
        
        
def get_run_group(task_name: str, do_tune: bool=False, group: str=None, cur_time: str=None):
    """
    Get wandb run group. If time is provided, will group all tasks under the same time group
    """
    run_group = "tune" + "_" if do_tune else "" # if hyperapram tuning, add tune to group name
    if cur_time is None:
        run_group += task_name + "_"
        
    run_group += group + "_" if group not in [None, ""] else ""
    run_group += time.strftime("%m-%d-%H", time.localtime()) if cur_time is None else cur_time 
    return run_group


class MyAwesomeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.large_lr = kwargs.pop("large_lr", False)
        self.use_scaler = kwargs.pop("use_scaler", False)
        self.new_lr = kwargs.pop("new_lr", 5e-3)
        super().__init__(*args, **kwargs)
        if hasattr(self.model, "roberta") and self.train_dataset is not None:
            self.model.roberta.trainer = self # for re-initializing optimizer to add monarch params
            len_dataloader = len(self.get_train_dataloader()) // self.args.gradient_accumulation_steps
            self.num_training_steps = math.ceil(len_dataloader * self.args.num_train_epochs)
            
    # def evaluate(self, eval_dataset=None):
    #     output = super().evaluate(eval_dataset)
    #     # check if ray tune is on
    #     if tune.is_session_enabled():
    #         tune.report(**output)
    #     return output
    
    def training_step(self, model, inputs):
        # for re-initializing optimizer to add monarch params
        # assignment is faster than if; just do it every time. Must assign like this in ray tune
        self.model.roberta.trainer = self 
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
                    logger.info(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        return self.optimizer

    # def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    #     """
    #     Added num_cycles to support cosine_with_restarts
    #     """
    #     if self.lr_scheduler is None:
    #         self.lr_scheduler = get_scheduler(
    #             self.args.lr_scheduler_type,
    #             optimizer=self.optimizer if optimizer is None else optimizer,
    #             num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
    #             num_training_steps=num_training_steps,
    #             scheduler_specific_kwargs={"num_cycles": 3} if self.args.lr_scheduler_type == "cosine_with_restarts" else None
    #         )
    #     return self.lr_scheduler


    # def create_optmizer_and_scheduler(self, num_training_steps):
    #     super().create_optimizer_and_scheduler(num_training_steps)
    #     self.num_training_steps = num_training_steps
    
    
##################################### Task configs #####################################

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    