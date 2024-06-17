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
    echo "Using flags ${FLAGS[@]} from the command line."
else
    echo "Using no additional flags."
fi


time=$(date "+%m-%d-%H")
# python run_glue.py /fly/task_configs/monarch_roberta_glue/cola.json   --time=$time "${FLAGS[@]}" &
# wait for the previous job to occupy GPU memory to pick the correct vacant device for the next
# sleep 20; python run_glue.py /fly/task_configs/monarch_roberta_glue/mrpc.json   --time=$time "${FLAGS[@]}" 
sleep 20; python run_glue.py /fly/task_configs/monarch_roberta_glue/qnli.json   --time=$time "${FLAGS[@]}" &
# sleep 20; python run_glue.py /fly/task_configs/monarch_roberta_glue/rte.json    --time=$time "${FLAGS[@]}" 
# sleep 20; python run_glue.py /fly/task_configs/monarch_roberta_glue/sst-2.json  --time=$time "${FLAGS[@]}"& 
sleep 20; python run_glue.py /fly/task_configs/monarch_roberta_glue/stsb.json   --time=$time "${FLAGS[@]}"

sleep 20; python run_glue.py /fly/task_configs/monarch_roberta_glue/qqp.json    --time=$time "${FLAGS[@]}" &
sleep 20; python run_glue.py /fly/task_configs/monarch_roberta_glue/mnli.json   --time=$time "${FLAGS[@]}" 

out_path="results/monarch_monarch_roberta_glue"
# find all .tsv files in results/monarch_monarch_roberta_glue and zip them
find results/monarch_monarch_roberta_glue -maxdepth 2 -name "*.tsv" | zip -j results/monarch_monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in $out_path to glue_submit.zip. Ready for submission."
