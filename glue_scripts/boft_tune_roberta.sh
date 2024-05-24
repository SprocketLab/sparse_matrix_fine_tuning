#!/bin/sh
time=$(date "+%m-%d-%H")
for task in cola mrpc mnli rte sst-2 stsb qnli qqp
do
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_glue.py task_configs/boft_roberta_glue/${task}.json --use_monarch=False --use_boft=True --use_wandb=True --do_tune=True --project="BOFT_GLUE_tune" --notes="roberta" --time=$time  
done