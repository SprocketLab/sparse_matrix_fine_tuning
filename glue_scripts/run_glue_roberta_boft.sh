#!/bin/sh
time=$(date "+%m-%d-%H")
for task in mrpc mnli rte sst-2 stsb qnli qqp
do
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_glue.py task_configs/glue_deberta/${task}.json --use_monarch=False --use_boft=True --use_wandb=True --project="BOFT_GLUE" --notes="roberta" --time=$time  
done