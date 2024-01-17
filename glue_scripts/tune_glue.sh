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

# Run Python scripts with correctly formatted flags
# python run_glue.py /fly/task_configs/glue_peft_configs/cola.json --do_tune=True --project="monarch_glue_tune" "${FLAGS[@]}"   
python run_glue.py /fly/task_configs/glue_peft_configs/mrpc.json --do_tune=True --project="monarch_glue_tune" "${FLAGS[@]}"   
python run_glue.py /fly/task_configs/glue_peft_configs/qnli.json --do_tune=True --project="monarch_glue_tune" "${FLAGS[@]}"   
python run_glue.py /fly/task_configs/glue_peft_configs/rte.json  --do_tune=True --project="monarch_glue_tune" "${FLAGS[@]}"   
python run_glue.py /fly/task_configs/glue_peft_configs/sst-2.json --do_tune=True --project="monarch_glue_tune" "${FLAGS[@]}"  
python run_glue.py /fly/task_configs/glue_peft_configs/stsb.json --do_tune=True --project="monarch_glue_tune" "${FLAGS[@]}"   
python run_glue.py /fly/task_configs/glue_peft_configs/qqp.json  --do_tune=True --project="monarch_glue_tune" "${FLAGS[@]}"   
python run_glue.py /fly/task_configs/glue_peft_configs/mnli.json --do_tune=True --project="monarch_glue_tune" "${FLAGS[@]}"   
python run_glue.py /fly/task_configs/glue_peft_configs/wnli.json --do_tune=True --project="monarch_glue_tune" "${FLAGS[@]}"  

find . -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in $out_path to glue_submit.zip. Ready for submission."