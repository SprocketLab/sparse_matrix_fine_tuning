import os
import sys
sys.path.append("/fly")
sys.path.append("/fly/pyreft/pyvene")
sys.path.append("/fly/pyreft")
os.environ["PYTHONPATH"] = "/fly:/fly/pyreft:/fly/pyreft/pyreft/:/fly/pyreft/pyvene" # Allow ray tune to copy dependencies
from train_utils import *
import torch
import argparse
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
    TrainingArguments,
    RobertaForSequenceClassification
)
from accelerate import load_checkpoint_and_dispatch
from transformers.trainer_utils import EvalPrediction
import wandb
import evaluate
import datetime
import json
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import Dataset

from task_config import task_config
from dataset import LoReftGLUEDataset, LoReftSupervisedDataset
from compute_metrics import compute_metrics
from copy import deepcopy
from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    ReftTrainerForSequenceClassification,
    NoreftIntervention,
    LoreftIntervention,
    NoIntervention,
    MoReIntervention,
    ReftDataCollator,
    ReftModel
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler


args = None
tokenizer = None
reft_model = None
best_hyperparams = None
config = None
device = "cuda" if torch.cuda.is_available() else "cpu"
classification_tasks = {"glue"}
residual_stream_component_mapping = {
    "robertaformaskedlm": "roberta.encoder.layer[%s].output"
}
dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}
peft_config = json.load(open("/fly/task_configs/llama_mmlu/peft_config.json", "r"))

def model_init(hyperparams: dict = best_hyperparams):
    global peft_config, args, tokenizer, reft_model, config
    if hyperparams == None:
        hyperparams = {}
        
    # everything is guarded by a single seed
    set_seed(args.seed)
    dtype = dtype_mapping[args.dtype]
    model_name = args.model
    # Hyperparameter search
    if args.blk_r != -1:
        hyperparams["blk_r"] = args.blk_r
    if args.nblocks != -1:
        hyperparams["nblocks"] = args.nblocks
    if hyperparams is not None:
        for k in peft_config.keys():
            if k in hyperparams.keys() and hyperparams[k] != peft_config[k]:
                print("Overriding {} = {} from best HP".format(k, hyperparams[k]))
                peft_config[k] = hyperparams[k]

    if wandb.run is not None:
        wandb.run.config.update(peft_config)
        wandb.run.config.update({"dtype": dtype})

    if reft_model is None:
        if args.task in classification_tasks:
            config = AutoConfig.from_pretrained(
                model_name, num_labels=args.num_labels,
                finetuning_task=args.train_dataset,
                load_in_8bit=True if args.dtype == "float8" else False,
                device_map=device
            )
            # full precision loading since usually for small models
            reft_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config, # just providing the label
                torch_dtype=dtype if args.dtype != "float8" else None,
                load_in_8bit=True if args.dtype == "float8" else False,
                device_map=device,
                max_memory={0: 0.8}, 
                # attn_implementation="flash_attention_2"
            )

        else:
            reft_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype if args.dtype != "float8" else None,  # save memory
                load_in_8bit=True if args.dtype == "float8" else False,
                device_map=device,
                max_memory={0: 0.8}, # device: memory
                attn_implementation="flash_attention_2"
            )
            config = reft_model.config
    
    if not isinstance(reft_model, ReftModel):
        # Optionally apply ReFT
        if args.intervention_type == "LoreftIntervention":
            intervention_type = LoreftIntervention
        elif args.intervention_type == "NoreftIntervention":
            intervention_type = NoreftIntervention
        elif args.intervention_type == "MoReIntervention":
            intervention_type = partial(MoReIntervention, dtype=dtype, blk_r=args.blk_r, nblocks=args.nblocks)
        else:
            intervention_type = NoIntervention
        
        # intervention config based on model type
        intervention_dtype = torch.bfloat16 if isinstance(dtype, str) else dtype
        model_arch = config.architectures[0].lower()
        if model_arch in residual_stream_component_mapping:
            representations = [{
                "component": residual_stream_component_mapping[model_arch] % l,
                "intervention": intervention_type(
                    embed_dim=config.hidden_size, low_rank_dimension=args.rank,
                    dropout=args.dropout, dtype=intervention_dtype, act_fn=args.act_fn, device=device,
                    add_bias=args.add_bias
                )
            } for l in args.layers]
            task_type=TaskType.SEQ_CLS
        else:
            representations = [{
                "layer": l, "component": "block_output",
                "low_rank_dimension": args.rank,
                "intervention": intervention_type(
                    embed_dim=config.hidden_size, low_rank_dimension=args.rank,
                    dropout=args.dropout, dtype=intervention_dtype, act_fn=args.act_fn, device=device,
                    add_bias=args.add_bias
                )
            } for l in args.layers]
            task_type=TaskType.CAUSAL_LM
    
        reft_model.enable_input_require_grads()
        reft_config = ReftConfig(representations=representations)
        reft_model = get_reft_model(reft_model, reft_config, set_device=not isinstance(dtype, str))
        # for GLUE tasks, we enable gradients on the classifier head.
        # the parameter will be counted as well.
        if args.task == "glue" and args.allow_cls_grad:
            for param in reft_model.model.classifier.parameters():
                # reft_model with HF trainer will automatically pick up these params to optimize
                param.requires_grad = True
        
        # Monarch adaptation
        if args.monarch:
            peft_config["dtype"] = dtype
            init_monarch_layers(reft_model, peft_config)

    param_stats(reft_model, training=False)
    reft_model.print_trainable_parameters()
    return reft_model
    
    
def finetune(
    act_fn: str,
    add_bias: bool,
    model: str,
    layers: str,
    rank: int,
    position: str,
    epochs: int,
    seed: int,
    intervention_type: str,
    max_n_train_example: int,
    max_n_eval_example: int,
    use_wandb: bool,
    wandb_name: str,
    gradient_accumulation_steps: int,
    batch_size: int,
    output_dir: str,
    task: str,
    lr: float,
    schedule: str,
    data_dir: str,
    train_dataset: str,
    eval_dataset: str,
    save_model: bool,
    eval_batch_size: int,
    warmup_ratio: float,
    weight_decay: float,
    dropout: float,
    test_split: str,
    train_on_inputs: bool,
    max_length: int,
    use_normalized_template: bool,
    allow_cls_grad: bool,
    metric_for_best_model: str,
    dtype: str,
    logging_steps: int,
    wandb_dir: str,
    wandb_proj: str,
    share_weights: bool,
    greedy_decoding: bool,
    temperature: float,
    top_p: float,
    top_k: float,
    **kwargs
):
    """
    Generic Representation Finetuning.
    """
    global tokenizer, reft_model, peft_config
    
    assert task in task_config
    if data_dir is not None:
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
    
    # store/log run details
    print(
        f"task: {task}, model: {model}, intervention_type: {intervention_type}, "
        f"layers: {layers}, rank: {rank}, "
        f"position: {position}, epoch: {epochs}, train_on_inputs: {train_on_inputs}, "
        f"max_length: {max_length}, allow_cls_grad: {allow_cls_grad}"
    )

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    model_str = args.model.split("/")[-1]
    if args.train_dataset is not None:
        run_name = f"{model_str}.{args.task}.{args.train_dataset}.{args.test_split}.{now}"
    else:
        run_name = f"{model_str}.{args.task}.{now}"
    # load dataset splits
    assert task in task_config, f"Unrecognized task: {task}"
    train_datasets = task_config[task]["train_datasets"] if train_dataset is None else [train_dataset]
    if task == "glue":
        eval_datasets = [train_dataset]
    else:
        eval_datasets = task_config[task]["eval_datasets"] if args.eval_dataset is None else [args.eval_dataset]
    
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            model_max_length=args.max_length,
            padding_side="right",
            use_fast=False,
        )
    tokenizer.pad_token = tokenizer.unk_token
    # which layers to intervene on
    if isinstance(args.layers, str):
        if args.layers != "all":
            args.layers = [int(l) for l in args.layers.split(";")]
        else:
            temp_config = AutoConfig.from_pretrained(args.model)
            args.layers = [l for l in range(temp_config.num_hidden_layers)]

        # position str takes the following formats:
        # f1 -> first token; f2 -> first two tokens.
        # f1+l1 -> first and last tokens; f2+l2 -> first and last two tokens.
        # fn or ln shares the same intervention.
        if "+" in args.position and not args.share_weights:
            args.layers += args.layers
        
    ReftDataset = LoReftGLUEDataset if task == "glue" else LoReftSupervisedDataset 
    path = os.path.join(data_dir, train_datasets[0]) if data_dir is not None else train_datasets[0]
    if not args.do_train:
        max_n_train_example = 1
    train_dataset = ReftDataset(
        task, train_datasets[0] if task == "glue" else path, 
        tokenizer, data_split="train", seed=seed, max_n_example=max_n_train_example,
        **{"num_interventions": len(args.layers), "position": position, 
           "share_weights": share_weights}
    )
    trigger_tokens = train_dataset.trigger_tokens
    args.num_labels = train_dataset.num_labels
    
    all_eval_datasets = {}
    for eval_dataset in eval_datasets:
        test_splits = test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            if args.do_tune: 
                split = "train" # TODO: Ensure eval_loop doesn't throw a bug.. need to change later
                
            path = os.path.join(data_dir, eval_dataset) if data_dir is not None else eval_dataset
            raw_eval = ReftDataset(
                task, eval_dataset if task == "glue" else path, 
                tokenizer, data_split=split, seed=seed, max_n_example=max_n_eval_example,
                **{"num_interventions": len(args.layers), "position": position, 
                   "share_weights": share_weights}, is_eval=True
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets
    # Initialize model
    reft_model = model_init()
    n_params = reft_model.count_parameters(include_model=False)
    
    if task == "glue":
        # we repartition the eval_datatsets into [1] 50% validation + [2] 50% test
        # we select the best model on [1] during training
        # we test the selected model on [2] to ensure fairness
        to_split_eval_datasets = eval_datasets[args.train_dataset][test_split][0]
        if len(to_split_eval_datasets) > 5000:
            in_train_n_eval_sample = 1000
        else:
            in_train_n_eval_sample = len(to_split_eval_datasets) // 2

        new_splits = torch.utils.data.random_split(
            to_split_eval_datasets, [len(to_split_eval_datasets)-in_train_n_eval_sample, in_train_n_eval_sample]
        )
        
        in_test_eval_datasets, in_train_eval_datasets = new_splits[0], new_splits[1]
        eval_datasets[args.train_dataset][test_split][0] = in_test_eval_datasets
        print("GLUE validation split (in training): ", len(in_train_eval_datasets))
        print("GLUE validation split (testing): ", len(eval_datasets[args.train_dataset][test_split][0]))

        is_regression = args.train_dataset == "stsb"
        metric = evaluate.load("glue", args.train_dataset)
        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def in_training_compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

        
    # select collator based on the type
    if task in classification_tasks:
        data_collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest"
        )
    else:
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest"
        )
        
    data_collator = ReftDataCollator(data_collator=data_collator_fn)

    # start wandb logging
    # if task == "tune_math": 
        # task = "math"
    # Now datasets are set up, use the actual task name
    if "tune" in task:
        task = task.split("_")[-1]
    task_dir = os.path.join(output_dir, task)
    output_dir = os.path.join(task_dir, args.group) if args.group else task_dir
    
    if args.notes:
        run_name = args.notes + "_" + run_name
        
    # Must set env variables to carry to them ray tune workers!
    if args.use_wandb == False:
        os.environ["WANDB_MODE"] = "offline"
    if args.resume:
        group_path = os.path.join(output_dir, "full_group.txt")
        if os.path.exists(group_path):
            os.environ["WANDB_RUN_GROUP"] = group = open(group_path, "r").read().strip()
    elif args.do_tune:
        os.environ["WANDB_RUN_GROUP"] = group = get_run_group(task, group=args.group, notes=args.notes, do_tune=True)
    else:
        group = None

    os.environ["WANDB_PROJECT"] = f"reft-monarch-{task}"
    run = wandb.init(
        project=os.environ["WANDB_PROJECT"], 
        name=run_name,
        dir=wandb_dir,
        group=group,
    )
    if not args.do_tune:
        watch_layers(reft_model.model)
    run.summary.update(vars(args))
    wandb.log(
        {"train/n_params": n_params})
    
    # # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        metric_for_best_model=metric_for_best_model if task == "glue" else None,
        # load_best_model_at_end=True if task == "glue" else False,
        logging_strategy="steps",
        save_total_limit=5, # for GLUE, it will save 2 at max.
        logging_steps=logging_steps,
        lr_scheduler_type=schedule,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        weight_decay=weight_decay,
        report_to="wandb" if use_wandb else "none",
        use_cpu=False if device == "cuda" else True,
        seed=seed,
        # until HF supports ReFT, this remains False! :)
        remove_unused_columns=False,
        do_eval=True
    )

    # make trainer
    trainer_class = ReftTrainerForSequenceClassification \
        if task in classification_tasks else ReftTrainerForCausalLM
    if task == "glue":
        eval_dataset = in_train_eval_datasets
    # Ray Tune requires eval. Otherwise the authors didn't use eval
    elif args.do_tune:
        assert len(eval_datasets) == 1, "Use only one eval set for HPO!"
        raw_eval = list(list(eval_datasets.values())[0].values())[0][0] # TODO: beautify
        eval_dataset = raw_eval
    elif task == "commonsense":
        eval_dataset = ReftDataset(
            task,  train_datasets[0], 
            tokenizer, data_split="train", seed=seed, max_n_example=max_n_eval_example,
            **{"num_interventions": len(args.layers), "position": position, 
                "share_weights": share_weights}, is_eval=True
        )
    else:
        eval_dataset = None
        training_args.evaluation_strategy = "no"
        
    trainer = trainer_class(
        model=reft_model,
        model_init=model_init,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=in_training_compute_metrics if task == "glue" else None,
    )
    # Eval more than once per epoch
    evals_per_epoch = args.evals_per_epoch
    trainer.args.eval_steps = len(train_dataset) \
        // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * evals_per_epoch)
        
    if args.do_tune:
        # Save full tune group name for resuming
        with open(os.path.join(training_args.output_dir, "full_group.txt"), "w") as f:
            f.write(group)
        
        _args = deepcopy(trainer.args)
        trainer.args.save_total_limit = 0 # Avoid flooding the disk during HPO
        trainer.args.load_best_model_at_end = False
        trainer.args.save_strategy = "no"
        trainer.args.evaluation_strategy = "steps" # Requried for Ray Tune
    
        ######################### Need to change for each task #########################
        # PEFT monarch search space
        if task == "math":
            real_bs = [16, 32, 64]
        elif task == "commonsense":
            real_bs = [16, 32, 64]
        else:
            raise NotImplementedError(f"Don't forget to manually pick bs for task {task} !")
        grad_acc_steps = [i // args.batch_size for i in real_bs]
        
        param_space = {
            # "nblocks": tune.choice(['sqrt(n)', 4]),
            "seed": training_args.seed,
            # "num_train_epochs": tune.choice([20, 25]),
            "learning_rate": tune.quniform(1e-4, 9e-4, 1e-4), 
            "gradient_accumulation_steps": tune.choice(grad_acc_steps), # Will OOM if tune batch size
            "weight_decay": tune.choice([0]),
            "lr_scheduler_type": tune.choice(["cosine", "linear"]), # mostly linear underperforms
            "dropout": tune.choice([0.05, 0.1]),
            "blk_r": peft_config["blk_r"],
            "nblocks": peft_config["nblocks"],
        }
        n_trials = args.n_trials 
        
        # Set up scheduler and reporter etc.
        metric = f'eval_loss'
        direction = "min" if "loss" in metric else "max" # minimize eval loss
        tune_unit = "iter"
        max_t = 40 * 60  if "tune_unit" == "time" else args.epochs * evals_per_epoch
        grade_period = 4 * 60  if tune_unit == "time" else 2
        time_attr = "time_total_s" if tune_unit == "time" else "training_iteration"
        ############################## End of task specific ##############################
        
        scheduler = ASHAScheduler(
            time_attr=time_attr,
            max_t=max_t,
            metric = metric,
            mode = direction,
            grace_period=grade_period,
        )
        # Do hyperparam optimization with Ray Tune
        best_run = trainer.hyperparameter_search(hp_space=lambda _: param_space,
            backend="ray",
            n_trials=n_trials, # under the hood it calls ray.tune.run(num_samples=n_trials, ...)
            scheduler=scheduler,
            keep_checkpoints_num=None,
            resources_per_trial={"cpu": 1, "gpu": 1},
            name=group,
            local_dir="/fly/ray_results",
            max_failures=9999, # tolerate OOM
            direction="maximize" if direction == "max" else "minimize",
            compute_objective=partial(get_hpo_metric, metric),
            resume=args.resume 
        )
        del trainer.model; model = None; free_memory() # Re-init model
        trainer.args = _args
        best_hyperparams = best_run.hyperparameters
        # Save the best HP for full training 
        print("Best hyperparameters: ", best_hyperparams)
        
        # Save hyperparams
        run_hp_path = os.path.join(training_args.output_dir, "best_hyperparams.json")
        task_hp_path = os.path.join(task_dir, "best_hyperparams.json")
        json.dump(best_hyperparams, open(run_hp_path, "w"))
        json.dump(best_hyperparams, open(task_hp_path, "w"))
    
    last_ckpt, _ = get_last_checkpoint(training_args.output_dir)
    # last_ckpt = os.path.join(last_ckpt, "intervenable_model")
    if args.do_train:
        load_best_hp(training_args.output_dir, task_dir)
        # TODO:enable resume
        if args.resume:
            trainer.train(resume_from_checkpoint=last_ckpt)
        else:
            trainer.train()
            
        # dump config
        args_dict = vars(args)
        args_dict["n_params"] = n_params
        json_file_name = f"{output_dir}/args.json"
        with open(json_file_name, 'w') as json_file:
            json.dump(args_dict, json_file, indent=4)

        # save model
        if save_model:
            reft_model.save(output_dir)
        # NOTE: force load best
        trainer._load_best_model()
    else:
        trainer.model.load_intervention(last_ckpt, include_model=True)
        
    # ensure everything is in eval mode
    reft_model.model.eval()
    for k,v in reft_model.interventions.items():
        _ = v[0].eval()

    print({"n_params": n_params})
    # do eval
    eval_results = {}
    for dataset_name in eval_datasets:
        # split evalset into chunks
        for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():
            
            generations, stats = compute_metrics(
                task, dataset_name, reft_model, tokenizer, eval_dataset, data_items,
                trigger_tokens, run_name, eval_batch_size, 
                data_collator if task in classification_tasks else None,
                split, greedy_decoding, temperature, top_p, top_k
            )

            # log
            eval_results.update(stats)
            if use_wandb:
                wandb.log(stats)
            generations = stats if generations is None else generations
            result_json_file_name = f"{output_dir}/{dataset_name}_{split}_outputs.json"
            with open(result_json_file_name, 'w') as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{output_dir}/eval_results.json"
    eval_results["n_params"] = n_params
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)
        

def main():
    parser = argparse.ArgumentParser(description="A simple script that takes different arguments.")
    
    parser.add_argument('-task', '--task', type=str, default=None)
    parser.add_argument('-data_dir', '--data_dir', type=str, default=None)
    parser.add_argument('-train_dataset', '--train_dataset', type=str, default=None)
    parser.add_argument('-eval_dataset', '--eval_dataset', type=str, default=None)
    parser.add_argument('-model', '--model', type=str, help='yahma/llama-7b-hf', default='yahma/llama-7b-hf')
    parser.add_argument('-seed', '--seed', type=int, help='42', default=42)
    parser.add_argument('-l', '--layers', type=str, help='2;10;18;26', default='2;10;18;26')
    parser.add_argument('-r', '--rank', type=int, help=8, default=8)
    parser.add_argument('-p', '--position', type=str, help='f1+l1', default='f1+l1')
    parser.add_argument('-e', '--epochs', type=int, help='1', default=1)
    parser.add_argument('-wandb', '--use_wandb', default=True, type=eval)
    parser.add_argument('-wandb_name', '--wandb_name', type=str, default="reft")
    parser.add_argument('-save_model', '--save_model', action='store_true')
    parser.add_argument('-max_n_train_example', '--max_n_train_example', type=int, default=None)
    parser.add_argument('-max_n_eval_example', '--max_n_eval_example', type=int, default=None)
    parser.add_argument(
        '-type', '--intervention_type', type=str, 
        help='LoreftIntervention', default="LoreftIntervention")
    parser.add_argument('-gradient_accumulation_steps', '--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=4)
    parser.add_argument('-eval_batch_size', '--eval_batch_size', type=int, default=4)
    parser.add_argument('-output_dir', '--output_dir', type=str, default="./official_results")
    parser.add_argument('-lr', '--lr', type=float, default=5e-3)
    parser.add_argument('-schedule', '--schedule', type=str, default='linear')
    parser.add_argument('-wu', '--warmup_ratio', type=float, default=0.00)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.00)
    parser.add_argument('-act_fn', '--act_fn', type=str, default=None)
    parser.add_argument('-add_bias', '--add_bias', action='store_true')
    parser.add_argument('-test_split', '--test_split', type=str, default="validation")
    parser.add_argument('-train_on_inputs', '--train_on_inputs', action='store_true')
    parser.add_argument('-max_length', '--max_length', type=int, help=512, default=512)
    parser.add_argument('-nt', '--use_normalized_template', action='store_true')
    parser.add_argument('-allow_cls_grad', '--allow_cls_grad', action='store_true')
    parser.add_argument('-metric_for_best_model', '--metric_for_best_model', type=str, default="accuracy")
    parser.add_argument('-dtype', '--dtype', type=str, default="bfloat16" if device == "cuda" else "float32")
    parser.add_argument('-logging_steps', '--logging_steps', type=int, help=1, default=1)
    parser.add_argument('-wandb_dir', '--wandb_dir', type=str, default='wandb')
    parser.add_argument('-wandb_proj', '--wandb_proj', type=str, default='MyReFT')
    parser.add_argument('-sw', '--share_weights', action='store_true')
    parser.add_argument('-gd', '--greedy_decoding', action='store_true')
    
    # Monarch
    parser.add_argument("--monarch", default=True, type=eval)
    parser.add_argument("--nblocks", default=-1, type=int)
    parser.add_argument("--blk_r", default=-1, type=int)
    parser.add_argument("--do_tune", action="store_true")
    parser.add_argument("--do_train", default=True, type=eval)
    
    # Ray Tune & wandb
    parser.add_argument("--n_trials", default=35, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evals_per_epoch", default=2, type=int)
    parser.add_argument("--group", type=str, default=None, help="Wandb run group for different model config etc.")
    # decoding params
    parser.add_argument('-t', '--temperature', type=float, default=None)
    parser.add_argument('-top_p', '--top_p', type=float, default=None)
    parser.add_argument('-top_k', '--top_k', type=float, default=None)

    parser.add_argument("--notes", type=str, default="")
    global args
    args = parser.parse_args()

    finetune(**vars(args))


if __name__ == "__main__":
    main()
