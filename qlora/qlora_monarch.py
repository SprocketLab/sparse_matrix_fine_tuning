# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
os.environ["PYTHONPATH"] = "/fly"
from os.path import exists, join
from dataclasses import dataclass, field
import sys
from copy import deepcopy
sys.path.append("/fly")
from train_utils import (
    param_stats,
    init_monarch_layers,
    get_hpo_metric,
    get_run_group,
    override_config,
    load_best_hp,
    watch_layers,
    set_merged,
    MySeq2SeqTrainer,
    get_last_checkpoint
)
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
from functools import partial
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import login
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate
import wandb
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from accelerate import load_checkpoint_and_dispatch

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import json
if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)
tokenizer = None
best_hyperparams = None
args = None
model = None
peft_config = json.load(open("/fly/task_configs/llama_mmlu/peft_config.json", "r"))

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b"
    )


@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    full_group: Optional[str] = field(
        default=None, metadata={"help": "Use the full group name for resuming HPO."}
    ) 
    resume: Optional[bool] = field(
        default=False, metadata={"help": "Resume HPO"}
    )
    n_trials: Optional[int] = field(
        default=20, metadata={"help": "Number of hyperparameter search trials."}
    )
    group: Optional[str] = field(
        default="", metadata={"help": "The wandb group name for the run."}
    )
    notes: Optional[str] = field(
        default="", metadata={"help": "Notes for the run."}
    )
    hf_token: Optional[str] = field(
        default=""
    )
    do_tune: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do Hyperparam tuning using ASHA."}
    )
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    max_memory_MB: int = field(
        default=38000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='wandb',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    use_wandb: bool = field(default=True)
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='adamw_torch', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=False, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def model_init(hyperparams: dict = best_hyperparams):
    global peft_config, model
    set_seed(args.seed)

    if hyperparams is not None:
        for k in peft_config.keys():
            if k in hyperparams.keys() and hyperparams[k] != peft_config[k]:
                print("Overriding {} = {} from best HP".format(k, hyperparams[k]))
                peft_config[k] = hyperparams[k]
    if torch.cuda.is_available():
        # n_gpus = torch.cuda.device_count()
        n_gpus = 1
    else:
        n_gpus = 0
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            device_map=device_map,
            max_memory=max_memory,
            # torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=True,
            use_auth_token=True,
            attn_implementation="flash_attention_2"
            # token=args.hf_token
        )
    model.enable_input_require_grads()
    # Monarch adaptation
    init_monarch_layers(model, peft_config)
    param_stats(model)
    watch_layers(model)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=False, # Fast tokenizer giving issues.
            tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
            trust_remote_code=True,
            use_auth_token=True,
        )
        if tokenizer._pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
            # LLaMA tokenizer may not have correct special tokens set.
            # Check and add them if missing to prevent them from being parsed into different tokens.
            # Note that these are present in the vocabulary.
            # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
            print('Adding special tokens.')
            tokenizer.add_special_tokens({
                    "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                    "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                    "unk_token": tokenizer.convert_ids_to_tokens(
                        model.config.pad_token_id if (model.config.pad_token_id != -1 and model.config.pad_token_id is not None) \
                            else tokenizer.pad_token_id
                    ),
            })

    # for name, module in model.named_modules():
        # if isinstance(module, LoraLayer):
        #     if args.bf16:
        #         module = module.to(torch.bfloat16)
        # if 'norm' in name:
            # module = module.to(torch.float32)
        # if 'lm_head' in name or 'embed_tokens' in name:
        #     if hasattr(module, 'weight'):
        #         if args.bf16 and module.weight.dtype == torch.float32:
        #             module = module.to(torch.bfloat16)
    return model


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )




def train():
    global args, model, peft_config, best_hyperparams
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    # Group by hpo runs 
    task_dir = args.output_dir
    if args.group is not None:
        args.output_dir = training_args.output_dir = os.path.join(args.output_dir, args.group) 
    print(f"output dir: {args.output_dir}")
    
    best_hyperparams = load_best_hp(args.output_dir, task_dir)
    if best_hyperparams is not None:
        override_config([best_hyperparams], extra_args)
    override_config([training_args, peft_config], best_hyperparams)
    override_config([training_args, peft_config], extra_args) # CLA args can override best hp
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    
    print(args)
    login(args.hf_token)
    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "offline"
        
    group = "mmlu"
    if args.do_tune:
        os.environ["WANDB_PROJECT"] = "llama-mmlu-tune"
        os.environ["WANDB_RUN_GROUP"] = get_run_group(group, args.do_tune, args.group, notes=args.notes)

    else:
        os.environ["WANDB_PROJECT"] = "llama-mmlu"
        os.environ["WANDB_RUN_GROUP"] = get_run_group(group, False, args.group, notes=args.notes)
    if args.resume:
        path = os.path.join(args.output_dir, "full_group.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                os.environ["WANDB_RUN_GROUP"] = f.read()
    wandb.init(project=os.environ["WANDB_PROJECT"], group=os.environ["WANDB_RUN_GROUP"], config=vars(args).update(peft_config)) 
    print(f"wandb group name: {os.environ['WANDB_RUN_GROUP']}")

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    _ = model_init(vars(args))
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    global tokenizer

    model.config.use_cache = False
    print('loaded model')

    data_module = make_data_module(tokenizer=tokenizer, args=args)
    trainer = MySeq2SeqTrainer(
        model_init=model_init,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )
    # Callbacks
    # if not args.full_finetune:
    #     trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")
        class MMLUEvalCallback(transformers.TrainerCallback):
            @torch.no_grad()
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds']
                    )['accuracy']
                    results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                    subject_scores.append(subject_score)
                
                # Average acc over all tasks?
                results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores) 
                trainer.log(results)
                # save results locally
                json.dump(results, open(os.path.join(args.output_dir, f'mmlu_{args.mmlu_split}_results.json'), 'w'))
                trainer.data_collator.source_max_len = source_max_len

        # trainer.add_callback(MMLUEvalCallback)

    all_metrics = {"run_name": args.run_name}
    
    if args.do_tune:
        # Save full tune group name for resuming
        with open(os.path.join(training_args.output_dir, "full_group.txt"), "w") as f:
            f.write(os.environ["WANDB_RUN_GROUP"])
        
        _args = deepcopy(trainer.args)
        
        # Avoid flooding the disk during HPO
        trainer.args.save_total_limit = 0 
        trainer.args.load_best_model_at_end = False
        trainer.args.save_strategy = "no"
        
        # PEFT monarch search space
        param_space = {
            # "nblocks": tune.choice(['sqrt(n)', 4]),
            "seed": training_args.seed,
            # "num_train_epochs": tune.choice([20, 25]),
            "learning_rate": tune.quniform(8e-5, 6e-4, 2e-5), 
            "gradient_accumulation_steps": tune.choice([16, 32]), # Will OOM if tune batch size
            "weight_decay": tune.choice([0]),
            "lr_scheduler_type": tune.choice(["cosine", "linear"]), # mostly linear underperforms
            "blk_r": peft_config["blk_r"],
            "nblocks": peft_config["nblocks"],
        }
        n_trials = args.n_trials 
        
        # Set up scheduler and reporter etc.
        direction = "min"
        tune_unit = "iter"
        max_t = 40 * 60  if "tune_unit" == "time" else 7
        metric = f'eval_loss'
        grade_period = 4 * 60  if tune_unit == "time" else 2
        time_attr = "time_total_s" if tune_unit == "time" else "training_iteration"
        
        scheduler = ASHAScheduler(
            time_attr=time_attr,
            max_t=max_t,
            metric = metric,
            mode = direction,
            grace_period=grade_period,
        )
        reporter = CLIReporter(
            parameter_columns=["learning_rate", "per_device_train_batch_size", "weight_decay"],
            metric_columns=["train_loss", "eval_loss", metric, "training_iteration"],
            max_progress_rows=9,
            max_report_frequency=9,
        )   
        # Do hyperparam optimization with Ray Tune
        best_run = trainer.hyperparameter_search(
            hp_space=lambda _: param_space,
            backend="ray",
            n_trials=n_trials, # under the hood it calls ray.tune.run(num_samples=n_trials, ...)
            scheduler=scheduler,
            keep_checkpoints_num=None,
            resources_per_trial={"cpu": 1, "gpu": 1},
            name=os.environ["WANDB_RUN_GROUP"],
            local_dir="/fly/ray_results",
            max_failures=9999, # tolerate OOM
            direction="maximize" if direction == "max" else "minimize",
            compute_objective=partial(get_hpo_metric, metric),
            resume=args.resume 
        )
        trainer.args = _args
        best_hyperparams = best_run.hyperparameters
            
        # Save the best HP for full training 
        print("Best hyperparameters: ", best_hyperparams)
        # Save in the run dir
        cur_tune_path = os.path.join(training_args.output_dir, "best_hyperparams.json")
        json.dump(best_hyperparams, open(cur_tune_path, "w"))

    
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        ckpt = checkpoint_dir if args.resume else checkpoint_dir
        train_result = trainer.train(resume_from_checkpoint=ckpt)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
        

    # Evaluation
    if args.do_eval:
        if args.do_train:
            trainer._load_best_model()
        else:
            last_checkpoint, _ = get_last_checkpoint(args.output_dir)
            print(f"Loading checkpoint from {last_checkpoint}")
            load_checkpoint_and_dispatch(trainer.model, last_checkpoint) 
        # NOTE: to avoid merging monarch weights twice. Removed due to using 'MySeq2SeqTrainer'
        
        # 1. models are often saved in safetensors instead of pickle, which don't store variables or code like self.merged
        # 2. load_checkpoint_and_dispatch actually loads the merged state dict; we shouldn't re-merge it.
        # We would NOT need this if we only save the monarch weights, not the merged dense weights
        # set_merged(trainer.model)
        
        logger.info("*** Evaluate ***")
        trainer.add_callback(MMLUEvalCallback)
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()
