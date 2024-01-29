#!/bin/bash

# Initialize an array for flags
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

CUDA_VISIBLE_DEVICES=0 python run_glue.py /fly/task_configs/glue_peft_configs/cola.json   "${FLAGS[@]}" &
CUDA_VISIBLE_DEVICES=1 python run_glue.py /fly/task_configs/glue_peft_configs/mrpc.json   "${FLAGS[@]}" &
CUDA_VISIBLE_DEVICES=3 python run_glue.py /fly/task_configs/glue_peft_configs/qnli.json   "${FLAGS[@]}" &
CUDA_VISIBLE_DEVICES=3 python run_glue.py /fly/task_configs/glue_peft_configs/rte.json    "${FLAGS[@]}" &
CUDA_VISIBLE_DEVICES=5 python run_glue.py /fly/task_configs/glue_peft_configs/sst-2.json  "${FLAGS[@]}" &
CUDA_VISIBLE_DEVICES=6 python run_glue.py /fly/task_configs/glue_peft_configs/stsb.json   "${FLAGS[@]}" &
CUDA_VISIBLE_DEVICES=7 python run_glue.py /fly/task_configs/glue_peft_configs/qqp.json    "${FLAGS[@]}" &
CUDA_VISIBLE_DEVICES=0 python run_glue.py /fly/task_configs/glue_peft_configs/mnli.json   "${FLAGS[@]}" 
CUDA_VISIBLE_DEVICES=3  python run_glue.py /fly/task_configs/glue_peft_configs/wnli.json  "${FLAGS[@]}"

$out_path = "results/monarch_roberta_glue"
# find all .tsv files in results/monarch_roberta_glue and zip them
find results/monarch_roberta_glue -maxdepth 2 -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in $out_path to glue_submit.zip. Ready for submission."