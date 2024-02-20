#!/bin/bash
# Initialize an array for flags
source chtc_job_scripts/.env
FLAGS=()

# Loop through all the arguments
for arg in "$@"
do
    # Check if the argument is in --key=value format
    if [[ $arg == --* ]]; then
        # Append the argument to FLAGS array, preserving the quoting
        FLAGS+=("$arg")
    fi
done

# Check if FLAGS array is not empty
if [ ${#FLAGS[@]} -ne 0 ]; then
    echo "Using flags ${FLAGS[@]}, from command line."
else
    echo "Using no additional flags."
fi

# For moving copied inputs to cwd in CHTC jobs
if [ -d "./sparse_matrix_finetuning" ]; then
    mv ./sparse_matrix_finetuning/* .
fi

time=$(date "+%m-%d-%H")
CUDA_VISIBLE_DEVICES=0 python run_glue.py /fly/task_configs/glue_peft_configs/cola.json   --time=$time "${FLAGS[@]}" &
CUDA_VISIBLE_DEVICES=1 python run_glue.py /fly/task_configs/glue_peft_configs/qnli.json   --time=$time "${FLAGS[@]}" &
# CUDA_VISIBLE_DEVICES=3 python run_glue.py /fly/task_configs/glue_peft_configs/rte.json    --time=$time "${FLAGS[@]}" &
# CUDA_VISIBLE_DEVICES=3 python run_glue.py /fly/task_configs/glue_peft_configs/sst-2.json  --time=$time "${FLAGS[@]}" 
CUDA_VISIBLE_DEVICES=3 python run_glue.py /fly/task_configs/glue_peft_configs/stsb.json   --time=$time "${FLAGS[@]}" 
CUDA_VISIBLE_DEVICES=3 python run_glue.py /fly/task_configs/glue_peft_configs/mrpc.json   --time=$time "${FLAGS[@]}" 
# CUDA_VISIBLE_DEVICES=4 python run_glue.py /fly/task_configs/glue_peft_configs/qqp.json    --time=$time "${FLAGS[@]}" 
# CUDA_VISIBLE_DEVICES=1 python run_glue.py /fly/task_configs/glue_peft_configs/mnli.json   --time=$time "${FLAGS[@]}" 
# CUDA_VISIBLE_DEVICES=3  python run_glue.py /fly/task_configs/glue_peft_configs/wnli.json  --time=$time "${FLAGS[@]}"

out_path="results/monarch_roberta_glue"
# find all .tsv files in results/monarch_roberta_glue and zip them
find results/monarch_roberta_glue -maxdepth 2 -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in $out_path to glue_submit.zip. Ready for submission."