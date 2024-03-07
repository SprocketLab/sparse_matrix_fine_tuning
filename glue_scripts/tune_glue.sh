#!/bin/bash
# Initialize an array for flags. This can override preceding args with the same name.
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

run_conditionally() {    
    # Check the last exit status
    if [ $? -ne 0 ]; then
        # If previous command fails, return 1
        return 1
    else
        # If previous command succeeds, run this command
        "$@"
    fi
}


# Run Python scripts with correctly formatted flags
# Run HP tuning in foreground while full training in background in parallel
python run_glue.py /fly/task_configs/glue_peft_configs/cola.json --project="monarch_glue_tune" --do_tune=True --do_train=False --do_eval=True --do_test=False "${FLAGS[@]}";
run_conditionally  python run_glue.py /fly/task_configs/glue_peft_configs/cola.json --project="monarch_glue_tune" --load_group=True "${FLAGS[@]}" &

python run_glue.py /fly/task_configs/glue_peft_configs/mrpc.json --project="monarch_glue_tune" --do_tune=True --do_train=False --do_eval=True --do_test=False "${FLAGS[@]}";
run_conditionally  psython run_glue.py /fly/task_configs/glue_peft_configs/mrpc.json --project="monarch_glue_tune" --load_group=True "${FLAGS[@]}" &

# python run_glue.py /fly/task_configs/glue_peft_configs/qnli.json --project="monarch_glue_tune" --do_tune=True --do_train=False --do_eval=True --do_test=False "${FLAGS[@]}";
# run_conditionally  psython run_glue.py /fly/task_configs/glue_peft_configs/qnli.json --project="monarch_glue_tune" --load_group=True "${FLAGS[@]}" &


# python run_glue.py /fly/task_configs/glue_peft_configs/rte.json --project="monarch_glue_tune" --do_tune=True --do_train=False --do_eval=True --do_test=False "${FLAGS[@]}";
# run_conditionally  python run_glue.py /fly/task_configs/glue_peft_configs/rte.json --project="monarch_glue_tune" --load_group=True "${FLAGS[@]}" &

# python run_glue.py /fly/task_configs/glue_peft_configs/stsb.json --project="monarch_glue_tune" --do_tune=True --do_train=False --do_eval=True --do_test=False "${FLAGS[@]}";
# run_conditionally  python run_glue.py /fly/task_configs/glue_peft_configs/stsb.json --project="monarch_glue_tune" --load_group=True "${FLAGS[@]}" &

# python run_glue.py /fly/task_configs/glue_peft_configs/sst-2.json --project="monarch_glue_tune" --do_tune=True --do_train=False --do_eval=True --do_test=False "${FLAGS[@]}";
# run_conditionally  python run_glue.py /fly/task_configs/glue_peft_configs/sst-2.json --project="monarch_glue_tune" --load_group=True "${FLAGS[@]}" &

# python run_glue.py /fly/task_configs/glue_peft_configs/qqp.json --project="monarch_glue_tune" --do_tune=True --do_train=False --do_eval=True --do_test=False "${FLAGS[@]}";
# run_conditionally  python run_glue.py /fly/task_configs/glue_peft_configs/qqp.json --project="monarch_glue_tune" --load_group=True "${FLAGS[@]}" &

# python run_glue.py /fly/task_configs/glue_peft_configs/mnli.json --project="monarch_glue_tune" --do_tune=True --do_train=False --do_eval=True --do_test=False "${FLAGS[@]}";
# run_conditionally  python run_glue.py /fly/task_configs/glue_peft_configs/mnli.json --project="monarch_glue_tune" --load_group=True "${FLAGS[@]}" &


# Zip files
find results/monarch_roberta_glue -maxdepth 2 -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files to glue_submit.zip. Ready for submission."