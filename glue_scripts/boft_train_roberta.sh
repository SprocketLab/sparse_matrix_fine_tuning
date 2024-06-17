#!/bin/sh

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
    echo "Using flags ${FLAGS[@]} from the command line."
else
    echo "Using no additional flags."
fi


time=$(date "+%m-%d-%H")
# python run_glue.py /fly/task_configs/boft_roberta_glue/cola.json   --time=$time --use_monarch=False --use_boft=True --project="BOFT_GLUE" "${FLAGS[@]}" &
# wait for the previous job to occupy GPU memory to pick the correct vacant device for the next
# sleep 20; python run_glue.py /fly/task_configs/boft_roberta_glue/mrpc.json   --time=$time --use_monarch=False --use_boft=True --project="BOFT_GLUE" "${FLAGS[@]}" 
sleep 20; python run_glue.py /fly/task_configs/boft_roberta_glue/qnli.json   --time=$time --use_monarch=False --use_boft=True --project="BOFT_GLUE" "${FLAGS[@]}" &
# sleep 20; python run_glue.py /fly/task_configs/boft_roberta_glue/rte.json    --time=$time --use_monarch=False --use_boft=True --project="BOFT_GLUE" "${FLAGS[@]}" 
sleep 20; python run_glue.py /fly/task_configs/boft_roberta_glue/sst-2.json  --time=$time --use_monarch=False --use_boft=True --project="BOFT_GLUE" "${FLAGS[@]}"& 
# sleep 20; python run_glue.py /fly/task_configs/boft_roberta_glue/stsb.json   --time=$time --use_monarch=False --use_boft=True --project="BOFT_GLUE" "${FLAGS[@]}"

# sleep 20; python run_glue.py /fly/task_configs/boft_roberta_glue/qqp.json    --time=$time --use_monarch=False --use_boft=True --project="BOFT_GLUE" "${FLAGS[@]}" &
sleep 20; python run_glue.py /fly/task_configs/boft_roberta_glue/mnli.json   --time=$time --use_monarch=False --use_boft=True --project="BOFT_GLUE" "${FLAGS[@]}" 