
""" 
Finetuning the library models for sequence classification on GLUE.
Ex. training usage: python run_glue.py /fly/task_configs/glue_peft_configs/cola.json 
Ex. Hyperparameter tuning usage: python run_glue.py /fly/task_configs/glue_peft_configs/cola.json --do_tune=True
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os, sys
import pynvml

def select_gpu(exclude=[]):
    """
    Select the GPU with maximum free memory
    """
    pynvml.nvmlInit()
    
    num_gpus = pynvml.nvmlDeviceGetCount()
    max_mem = 0
    max_gpu = 0
    if num_gpus == 0:
        raise Exception("No GPU found")
    
    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = mem_info.free
        
        if free_mem > max_mem and i not in exclude:
            max_mem = free_mem  
            max_gpu = i
            
    pynvml.nvmlShutdown()
    
    print("Selected GPU:", max_gpu, "with max memory %.2f GB" % (max_mem / 1024 ** 3))
    return max_gpu

# A bit ugly...but this only works before all torch libs are imported
if not "--do_tune=True" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(select_gpu())

from functools import partial
import json
import time
import logging
import random
import copy
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention, RobertaIntermediate
from src.models.modeling_roberta import RobertaForSequenceClassification
from src.hf_setup import (
    setup_logging_ckpt,
    DataTrainingArguments,
    ModelArguments,
    task_to_keys
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from train_utils import (
    override_config,
    MyAwesomeTrainer,
    get_run_group,
    parse_args,
    init_monarch_layers,
    get_hpo_metric,
    PEFT_ROBERTA_PATH,
    print_dtypes,
    load_best_hp
)

import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Ensure reproducibility given the same hardware
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
_DTYPES_PRINTED = False
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

best_hyperparams: dict = None
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
    "mrpc": "eval_accuracy",
    "qnli": "eval_accuracy",
    "qqp": "eval_accuracy",
    "rte": "eval_accuracy",
    "sst2": "eval_accuracy",
    "stsb": "eval_pearson",
    "wnli": "eval_accuracy",
}
logger = logging.getLogger(__name__)

def override_dict(dict_new, dict_old):
    if dict_new is not None:
        for k in dict_old.keys():
            if k in dict_new.keys() and dict_new[k] != dict_old[k]:
                print("Overriding {} = {} from best HP".format(k, dict_new[k]))
                dict_old[k] = dict_new[k]

def main(config: dict = None):
    ############################## Command line args ##############################
    args = parse_args()
    peft_config = json.load(open(PEFT_ROBERTA_PATH, "r"))  # load monarch config
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args.config_path))
    
    # NOTE: Extra args can override all training configs (best HP, peft_config, etc.)
    extra_args = override_config([model_args, data_args, training_args, peft_config], sys.argv[2:])
    use_monarch = args.use_monarch
    do_tune = args.do_tune
    use_wandb = args.use_wandb
    adapter = args.adapter
    tune_unit = args.tune_unit
    # For grouping runs in wandb
    group = args.group
    project = args.project
    full_group = args.full_group
    
    # Adapter settings
    if peft_config["q_v"]:
        peft_config["layers_to_adapt"] = ["query", "value"]
    if peft_config["mlp"]:
        peft_config["layers_to_adapt"] += ["dense"]
        
    # Do NOT use loss as saving metric, maximize eval metric instead
    training_args.metric_for_best_model = task_to_metric[data_args.task_name] 
    training_args.greater_is_better = True 
    training_args.optim = "adamw_torch" # forward compatibility
    
    # Set up wandb 
    task_name = data_args.task_name if do_tune else "" # Put all tasks in one group in standalone training
    os.environ["WANDB_RUN_GROUP"] = get_run_group(task_name, do_tune, group, args.time, args.notes) if not full_group else full_group
    # Upload host machine to wandb for locating ckpts
    if os.path.exists("hostname.txt"):
        hostname = open("hostname.txt", "r").readline().strip()
        os.environ["WANDB_HOST"] = hostname
    elif use_wandb:
        logging.warning("Try adding a hostname.txt (hostname > hostname.txt), or wandb will use random id from docker.")
        
        
    if full_group:
        group = ("_").join(full_group.split("_")[1:-1] if not full_group.startswith("tune") else full_group.split("_")[2:-1])
        
    if do_tune:
        assert tune_unit in ["time", "eval_iter"], "max_t (resources) must be either time or eval iteration"
        print("Tuning hyperparameters for", data_args.task_name)
    else:
        print("Full training for", data_args.task_name)
        
    if not use_wandb:
        print("Disabling wandb")
        os.environ["WANDB_MODE"] = "disabled"
        
    task_output_dir = os.path.join(training_args.output_dir, data_args.task_name)
    training_args.output_dir = os.path.join(task_output_dir, group) if group else os.path.join(task_output_dir, "default")
    os.makedirs(training_args.output_dir, exist_ok=True)

    # For resuming HPO    
    if args.resume_tune or args.load_group:
        path = os.path.join(training_args.output_dir, "full_group.txt")
        if os.path.exists(path):
            full_group = os.environ["WANDB_RUN_GROUP"] = open(path, "r").readline().strip()
            if "tune" in full_group:
                project = "monarch_glue_tune" 
            print("Loading wandb run group: ", os.environ["WANDB_RUN_GROUP"])
        else:
            logging.warning("No full_group.txt found in the output dir. Won't resume HPO/put this training run in the same wandb group.")
            args.resume_tune = args.load_group = False

    # Logging and checkpointing
    last_checkpoint = setup_logging_ckpt(training_args, logger, do_tune)
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
            label_list = json.load(open("/fly/task_configs/labels.json", "r"))[data_args.task_name]
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
    def model_init(hyperparams: dict = None):
        global best_hyperparams, _DTYPES_PRINTED
        set_seed(training_args.seed)
        if hyperparams is not None:
            best_hyperparams = hyperparams
        if training_args.fp16:
            dtype = torch.float16
        
        model = RobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
            
        if torch.cuda.is_available():
            model = model.to("cuda")    
        # For Hyperparameter search
        override_dict(best_hyperparams, peft_config)
        if not _DTYPES_PRINTED:
            print_dtypes(model)
            _DTYPES_PRINTED = True
        if use_monarch:
            # For legacy purposes, don't init here. I ran some experiments with 
            # modules initialized in the 1st training step, so init here instead would
            # break the reproducibility. TODO: Try ensuring nothing breaks the random seed
            # in between so that we can init here
            
            # init_monarch_layers(model.roberta, peft_config)
            model.roberta.init_monarch_layers = partial(init_monarch_layers, model.roberta)
            model.roberta.peft_config = peft_config
            
        # NOTE: Ray doesn't support torch.compile, plus a bug with trainer...
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
    if training_args.do_train or args.do_tune:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval or args.do_tune:
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
            print("result: ", result)
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

    """ @Wenxuan """
    # NOTE: 60K steps for QNLI， 80K for SST-2, 50K for MNLI
    if data_args.task_name == "qqp":
        training_args.eval_steps = 1250 # 110K total steps for QQP
        training_args.save_steps = 1250
    training_args.per_device_eval_batch_size = 64 
    
    # Initialize Trainer
    has_ckpt = any([file.startswith("checkpoint") for file in os.listdir(training_args.output_dir)])
    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint &= has_ckpt

    # Wandb config 
    if use_wandb:
        training_args.run_name = "glue_" + data_args.task_name # wandb run name
        os.environ["WANDB_PROJECT"] = "monarch_hf" + "_peft" if use_monarch else "GLUE_FT" 
        os.environ["WANDB_PROJECT"] = project if project else os.environ["WANDB_PROJECT"] # Override if provided
        
        # group runs within the same hour
        print("Wandb project: ", os.environ["WANDB_PROJECT"])
        print("Wandb run group: ", os.environ["WANDB_RUN_GROUP"])
    else:
        os.environ["WANDB_RUN_GROUP"] = os.environ["WANDB_PROJECT"] = "offline"
        
    if not adapter:
        peft_config["adapter"] = False # Will not use merging adapter style
    
    ############################ Ray Tune Hyperparameter optimization ############################
    if do_tune:
        # Save full tune group name for resuming
        with open(os.path.join(training_args.output_dir, "full_group.txt"), "w") as f:
            f.write(os.environ["WANDB_RUN_GROUP"])
            
        # clone args
        # Avoid flooding the disk during HPO
        tune_args = copy.deepcopy(training_args)
        tune_args.save_total_limit = 0 
        tune_args.load_best_model_at_end = False
        tune_args.save_strategy = "no"
        
        trainer = MyAwesomeTrainer(
            model_init=model_init,
            args=tune_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            large_lr=peft_config["large_lr"],
            new_lr=peft_config["new_lr"],
            use_scaler=peft_config["scaler"],
        )
        # PEFT monarch search space
        if use_monarch:
            param_space = {
                # "nblocks": tune.choice(['sqrt(n)', 4]),
                "seed": training_args.seed,
                # "large_lr": tune.sample_from(lambda _: np.random.uniform() > 0.4),
                "large_lr": False,
                # "num_train_epochs": tune.choice([20, 25]),
                "learning_rate": tune.quniform(1e-4, 6.6e-4, 2e-5), 
                "per_device_train_batch_size": tune.choice([16, 32]), # In Monarch-Mixer they mixed 32 and 16 
                "weight_decay": training_args.weight_decay,
                "lr_scheduler_type": "cosine", # mostly linear underperforms
                "blk_r": peft_config["blk_r"],
                "nblocks": peft_config["nblocks"],
            }
            n_trials = args.n_trials # 36 by default
                # block size = {256, 128, 64}
                # block rank = {4, 2, 8}
                # n_trials = 40
    
            if args.tune_blk_config:
                # TODO: Search a larger space, and fail the runs over the budget (~1.2M param)
                # Do NAS 
                param_space["blk_r"] = tune.choice([1, 2, 4, 8])
                param_space["blk_sz"] = tune.choice([64, 128, 512])
                param_space["lr_scheduler_type"] = "cosine"
                n_trials += 10
        else:
            # Vanilla finetuning
            param_space = {
                "learning_rate": tune.grid_search([1e-5, 2e-5, 3e-5]),
                "per_device_train_batch_size": tune.grid_search([16, 32]),
                "weight_decay": tune.choice([0.1]),
                "lr_scheduler_type": tune.grid_search(["cosine"]),
            }
            n_trials = 1 # grid search will try all combinations by default
        
        
        # Set up scheduler and reporter etc.
        direction = "max"
        max_t = 40 * 60 if tune_unit == "time" else 15 # mins or eval iterations
        if data_args.task_name == "mrpc":
            max_t = 30 * 60 if tune_unit == "time" else 12
        elif data_args.task_name == "stsb":
            max_t = 25 * 60 if tune_unit == "time" else 11
        elif data_args.task_name == "cola":
            max_t = 35 * 60 if tune_unit == "time" else 14
            
        grade_period = 4 * 60  if tune_unit == "time" else 3
        time_attr = "time_total_s" if tune_unit == "time" else "training_iteration"
        scheduler = ASHAScheduler(
            time_attr=time_attr,
            max_t=max_t,
            metric = task_to_metric[data_args.task_name],
            mode = direction,
            grace_period=grade_period,
        )
        reporter = CLIReporter(
            parameter_columns=["learning_rate", "per_device_train_batch_size", "weight_decay"],
            metric_columns=["train_loss", "eval_loss", task_to_metric[data_args.task_name], "training_iteration"],
            max_progress_rows=9,
            max_report_frequency=9,
        )   
        # Do hyperparam optimization with Ray Tune
        best_run = trainer.hyperparameter_search(
            hp_space=lambda _: param_space,
            backend="ray",
            n_trials=n_trials, # under the hood it calls ray.tune.run(num_samples=n_trials, ...)
            scheduler=scheduler,
            keep_checkpoints_num=0,
            checkpoint_score_attr="min-" + task_to_metric[data_args.task_name], # rank in decreasing order
            progress_reporter=reporter,
            resources_per_trial={"cpu": 1, "gpu": args.gpus_per_trial},
            local_dir="ray_results",
            name=os.environ["WANDB_RUN_GROUP"],
            max_failures=100, # tolerate OOM
            direction="maximize" if direction == "max" else "minimize",
            compute_objective=partial(get_hpo_metric, task_to_metric[data_args.task_name]),
            resume=args.resume_tune 
        )
        best_hp = best_run.hyperparameters
            
        # Save the best HP for full training 
        print("Best hyperparameters: ", best_hp)
        # Save in the run dir
        cur_tune_path = os.path.join(training_args.output_dir, "best_hyperparams.json")
        json.dump(best_hp, open(cur_tune_path, "w"))
        if args.as_base_hp or group == "":
            json.dump(best_hp, open(os.path.join(task_output_dir, "best_hyperparams.json"), "w"))
    
    
    ############################## Full training ##############################
    # load best hyperparams for the group
    best_param_path = os.path.join(training_args.output_dir, "best_hyperparams.json")
    base_param_path = os.path.join(task_output_dir, "best_hyperparams.json")
    if not os.path.exists(best_param_path):
        logging.warning("No hyperparams for this group found. Using best HP for this tasks' default config.\
                        This may be an unintended typo. (Check group name carefully)")
        best_param_path = base_param_path
        
    if os.path.exists(best_param_path):
        global best_hyperparams
        best_hyperparams = json.load(open(best_param_path, "r"))  
        print(f"Loading best hyperparams from {best_param_path}: ", best_hyperparams)
        override_config([best_hyperparams], sys.argv[2:])
        override_config([model_args, data_args, training_args], best_hyperparams)        
    else:
        best_hyperparams = None
        logging.warning("No best hyperparams from HPO found. Using LoRA HPs.")

    trainer = MyAwesomeTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        large_lr=peft_config["large_lr"],
        new_lr=peft_config["new_lr"],
        use_scaler=peft_config["scaler"],
    )

    # # Training
    if training_args.do_train and not do_tune:
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
    if training_args.do_eval and not do_tune:
        logger.info("*** Evaluate ***")
        last_checkpoint = os.path.join(get_last_checkpoint(training_args.output_dir),"pytorch_model.bin")
        override_dict(best_hyperparams, peft_config)
        trainer.model.roberta.init_monarch_layers()
        trainer.model.eval()
        trainer.model.load_state_dict(torch.load(last_checkpoint))
            
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

            output_predict_file = os.path.join(task_output_dir, f"{task_to_submit[task]}.tsv")
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
    
    print(f"Used best hyperparameters from {best_param_path}: ", best_hyperparams)
    print("peft_config: ", peft_config)
    
    # Attempt moving checkpoints to a more spacious disk
    # TODO: Debug
    # target_disk = "/data"
    # path = os.path.abspath(training_args.output_dir + "/**/pytorch_model.bin")
    # for ckpt in glob.glob(path, recursive=True):
    #     replace_with_symlink(ckpt, target_disk)
            
if __name__ == "__main__":
    main()
