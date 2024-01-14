
# @Wenxuan: Doesn't converge for some obscure reason..
""" Finetuning the library models for sequence classification on GLUE."""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import json
import transformers
import time
import logging
import os
import wandb
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
    Trainer
)
from fly_src.models.modeling_roberta import RobertaForSequenceClassification
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from train_utils import param_stats, override_config, MyAwesomeTrainer, get_run_group
import torch

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import ray.train.huggingface.transformers as ray_hf
from ray.tune import CLIReporter
from ray.air.integrations.wandb import WandbLoggerCallback

# NOTE: Given the same GPU (A100) results are mostly reproducible without this.
# Besides it does NOT gurarantee deterministic results across different GPUs
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

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
task_to_submit = {
    "cola": "CoLA",
    "mnli": "MNLI-m",
    "mnli-mm": "MNLI-mm",
    "mrpc": "MRPC",
    "qnli": "QNLI",
    "qqp": "QQP",
    "rte": "RTE",
    "sst2": "SST-2",
    "stsb": "STS-B",
    "wnli": "WNLI",
}

task_to_metric = {
    "cola": "eval_matthews_correlation",
    "mnli": "eval_accuracy",
    "mrpc": "eval_combined_score",
    "qnli": "eval_accuracy",
    "qqp": "eval_combined_score",
    "rte": "eval_accuracy",
    "sst2": "eval_accuracy",
    "stsb": "eval_pearson",
    "wnli": "eval_accuracy",
}
logger = logging.getLogger(__name__)


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


def main(config: dict = None):
    # See all possible arguments in fly_src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    peft_config = json.load(open("task_configs/glue_peft_configs/peft_monarch.json", "r"))  # load monarch config
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    # @Wenxuan
    # Example CLA usage: python run_glue_hf.py task_configs/cola.json 
    if sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        extra_args = override_config([model_args, data_args, training_args, peft_config], sys.argv[2:])
        
    elif config is not None:
        model_args, data_args, training_args = parser.parse_dict(config, allow_extra_keys=True)
        
        # allowing overriding peft config when passing in config with Ray Tune
        for k, v in config.items():
            if k in peft_config:
                peft_config[k] = v
    else:
        # parse command line args
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.metric_for_best_model = task_to_metric[data_args.task_name] # Do NOT use loss 
    training_args.greater_is_better = True 
    
    # NOTE: Accept extra command line args
    use_monarch = extra_args.get("monarch", True)
    do_tune = extra_args.get("do_tune", False)
    use_wandb = extra_args.get("use_wandb", True)
    # For grouping runs in wandb
    group = extra_args.get("group", None) 
    project = extra_args.get("project", None) 
    use_peft = extra_args.get("use_peft", True)
    
    if not use_wandb:
        print("Disabling wandb")
        os.environ["WANDB_MODE"] = "disabled"
            
    training_args.output_dir = os.path.join(training_args.output_dir, data_args.task_name) 
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = json.load(open("task_configs/labels.json", "r"))[data_args.task_name]
            # label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
            
    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    
    # helper to init and set hyperparams for Ray Tune search
    def model_init(hyperparams = None):
        torch.manual_seed(training_args.seed)
        model = RobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        
        # Hyperparameter search
        if hyperparams is not None:
            for k in peft_config.keys():
                if k in hyperparams.keys():
                    print("Overriding {} with {}".format(k, peft_config[k]))
                    peft_config[k] = hyperparams[k]

        if use_monarch:
            model.roberta.set_peft_config(peft_config)
            # model.roberta.init_monarch_layers()
        # NOTE: Ray doesn't support torch.compile and it also causes a bug with trainer...
        # if torch.__version__.startswith("2") and not do_tune:
        #     model = torch.compile(model)
        return model
    
    
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        config.label2id = {l: i for i, l in enumerate(label_list)}
        config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    # Get mapped datasets
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")
        
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # @Wenxuan
    # Initialize Trainer
    training_args.save_total_limit = 1 # avoid flooding the disk
    has_ckpt = any([file.startswith("checkpoint") for file in os.listdir(training_args.output_dir)])
    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint &= has_ckpt

    # Wandb config 
    if use_wandb:
        training_args.run_name = "glue_" + data_args.task_name # wandb run name
        os.environ["WANDB_PROJECT"] = "monarch_hf" + "_peft" if use_monarch else "GLUE_FT" 
        os.environ["WANDB_PROJECT"] = project if project else os.environ["WANDB_PROJECT"] # Override if provided
        
        # group runs within the same hour
        os.environ["WANDB_RUN_GROUP"] = get_run_group(data_args.task_name, do_tune, group)
        print("Wandb project: ", os.environ["WANDB_PROJECT"])
        print("Wandb run group: ", os.environ["WANDB_RUN_GROUP"])

    if not use_peft:
        peft_config["use_peft"] = False # Will not use merging adapter style
        
    
    # Ray Tune hyperparameter search
    if do_tune:
        trainer = MyAwesomeTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            same_lr=peft_config["same_lr"],
            use_scaler=peft_config["scaler"],
        )
        
        # PEFT monarch search space
        if use_monarch:
            param_space = {
                # "rank": tune.choice([1, 2, 3]),  # Tuning rank causes dim mismatch when merging
                # "nblocks": tune.choice(['sqrt(n)', 4]),
                "learning_rate": tune.quniform(8e-5, 2e-6, 1e-6),
                "per_device_train_batch_size": tune.choice([16, 32]), # In Monarch-Mixer they mixed 32 and 16 
                "weight_decay": tune.choice([0.01, 0.1, 1e-3]),
                "lr_scheduler_type": tune.choice(["cosine", "linear"]), # mostly linear underperforms
            }
            n_trials = 45
            
            if not use_peft:
                del param_space["nblocks"]
                n_trials = 36
                
        else:
            # Raw finetuning
            param_space = {
                "learning_rate": tune.grid_search([1e-5, 2e-5, 3e-5]),
                "per_device_train_batch_size": tune.grid_search([16, 32]),
                "weight_decay": tune.choice([0.1]),
                "lr_scheduler_type": tune.grid_search(["cosine", "linear"]),
            }
            n_trials = 1 # grid search will try all combinations by default
        
        direction = "max"
        scheduler = ASHAScheduler(
            max_t=12, # max_t * eval_every(eval_steps in configs) = max training steps
            metric = task_to_metric[data_args.task_name],
            mode = direction,
            grace_period=5,
        )
        
        reporter = CLIReporter(
            parameter_columns=["learning_rate", "per_device_train_batch_size", "weight_decay"],
            metric_columns=["train_loss", "eval_loss", task_to_metric[data_args.task_name], "training_iteration"],
            max_progress_rows=10,
            max_report_frequency=10,
        )   

        best_run = trainer.hyperparameter_search(
            hp_space=lambda _: param_space,
            backend="ray",
            n_trials=n_trials, # under the hood it calls ray.tune.run(num_samples=n_trials, ...)
            scheduler=scheduler,
            keep_checkpoints_num=0,
            checkpoint_score_attr="training_iteration",
            progress_reporter=reporter,
            resources_per_trial={"cpu": 1, "gpu": 0.5},
            local_dir="ray_results",
            name=os.environ["WANDB_RUN_GROUP"],
            max_failures=100, # tolerate OOM
            callbacks=[WandbLoggerCallback(project=os.environ["WANDB_PROJECT"], group=os.environ["WANDB_RUN_GROUP"])],
            direction="maximize" if direction == "max" else "minimize",
        )
    
        # Use best hyperparams for full training 
        print("Best hyperparameters: ", best_run.hyperparameters)
        best_param_path = os.path.join(training_args.output_dir, "best_hyperparams.json")
        json.dump(best_run.hyperparameters, open(best_param_path, "w"))
        do_tune = False # enable torch.compile
    
    # load best hyperparams from last tune 
    best_param_path = os.path.join(training_args.output_dir, "best_hyperparams.json")
    if os.path.exists(best_param_path):
        best_hyperparams = json.load(open(best_param_path, "r"))  
        print("Loading best hyperparams: ", best_hyperparams)
        override_config([best_hyperparams], sys.argv[2:])
        override_config([model_args, data_args, training_args], best_hyperparams)
    else:
        best_hyperparams = None
        
    if data_args.task_name == "qqp":
        training_args.eval_steps = 1000 # 110K total steps for QQP
        training_args.save_steps = 1000

    trainer = MyAwesomeTrainer(
        model=model_init(best_hyperparams),
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        same_lr=peft_config["same_lr"],
        use_scaler=peft_config["scaler"],
    )
    
    # # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if training_args.resume_from_checkpoint:
            last_checkpoint = os.path.join(get_last_checkpoint(training_args.output_dir),"pytorch_model.bin")
            trainer.model.roberta.init_monarch_layers()
            trainer.model.eval()
            trainer.model.load_state_dict(torch.load(last_checkpoint))
            # breakpoint()
            
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}
        
        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])
        
        t1 = time.time()
        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"{task_to_submit[task]}.tsv")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
        print("Inferece time on test set: ", time.time() - t1)

    print("Best hyperparameters: ", best_hyperparams)
    print("peft_config: ", peft_config)

if __name__ == "__main__":
    main()
