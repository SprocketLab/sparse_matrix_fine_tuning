# !/bin/bash

# Naive way to implement multiprocess training...
# Should use python mp spawn for gpu selection, job queueing, etc.
CUDA_VISIBLE_DEVICES=0 python run_glue_hf.py /fly/task_configs/cola.json &
CUDA_VISIBLE_DEVICES=1 python run_glue_hf.py /fly/task_configs/mrpc.json &
CUDA_VISIBLE_DEVICES=2 python run_glue_hf.py /fly/task_configs/qnli.json &
CUDA_VISIBLE_DEVICES=3 python run_glue_hf.py /fly/task_configs/rte.json  &
CUDA_VISIBLE_DEVICES=4 python run_glue_hf.py /fly/task_configs/sst-2.json &
CUDA_VISIBLE_DEVICES=5 python run_glue_hf.py /fly/task_configs/stsb.json &
CUDA_VISIBLE_DEVICES=6 python run_glue_hf.py /fly/task_configs/qqp.json  &
CUDA_VISIBLE_DEVICES=7 python run_glue_hf.py /fly/task_configs/mnli.json &
CUDA_VISIBLE_DEVICES=0  python run_glue_hf.py /fly/task_configs/wnli.json 

$out_path = "results/monarch_roberta_glue"
# find all .tsv files in results/monarch_roberta_glue and zip them
find . -name "*.tsv" | zip $out_path/glue_submit.zip -@
echo "Zipped all .tsv files in $out_path to glue_submit.zip. Ready for submission."