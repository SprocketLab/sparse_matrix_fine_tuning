#!/usr/bin/env python
# coding: utf-8
import os

# change cwd to last level
os.chdir(os.path.dirname(os.getcwd()))
import argparse
import logging

import torch
from transformers.trainer_utils import get_last_checkpoint

from train_utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--peft",
    action="store_true",
    help="""use PEFT
            (freeze weights, keep copy of new monarch matrices for each task) instead of FT""",
)
parser.add_argument(
    "--nblocks",
    type=int,
    default=4,
    help="""number of blocks in each block-diag monarch factor.
            If peft is false, will replace with sqrt(n) for best expressiveness """,
)
parser.add_argument(
    "--dataset",
    type=str,
    default="ag_news",
    help="""dataset to use""",
)
parser.add_argument("--prt_layers", action="store_true", help="print trainable layers for debugging")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logging.warning("No GPU available, using CPU will be super slow")


# ### Load dataset, model, and tokenizer
dataset_id = args.dataset
train_config = json.load(open(f"task_configs/{dataset_id}.json", "r"))  # load predefined hyperparams
if dataset_id != "ag_news":
    print(f"Training on {dataset_id} in GLUE")

model_id = train_config["model_id"]
del train_config["model_id"]
save_dir = f"./results/monarch_roberta_{dataset_id}"
os.makedirs(save_dir, exist_ok=True)

# set training config
peft_config = {
    "lora": False,
    "monarch": True,
    "rank": 1,
    "nblocks": 4,
    "layers_to_replace": ["query", "value"],
    "use_peft": args.peft,
    "precision": "fp16",
}
roberta_model, trainer, test_dataset, model_config = setup_trainer(
    model_id, dataset_id, save_dir, train_config, peft_config, device
)
roberta_model.train()
run_trainer(trainer, test_dataset, peft_config["precision"])


# Check loading fine-tuned Roberta
ckpt_dir = get_last_checkpoint(save_dir)
roberta_model = RobertaForSequenceClassification.from_pretrained(model_id, config=model_config).to(device)
roberta_model.roberta.set_peft_config(peft_config)
roberta_model.eval()
roberta_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "pytorch_model.bin")))
