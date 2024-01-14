# !/bin/bash

# Initialize an array for flags
FLAGS=()

# Loop through all the arguments
for arg in "$@"
do
    # Check if the argument is in --key=value format
    if [[ $arg == --* ]]; then
        # Append the argument to FLAGS array
        FLAGS+=("$arg")
    fi
done

# Function to join array elements
join_by() {
  local IFS="$1"
  shift
  echo "$*"
}

# Join the FLAGS array elements with a space
JOINED_FLAGS=$(join_by ' ' "${FLAGS[@]}")

# Check if JOINED_FLAGS is not empty
if [ -n "$JOINED_FLAGS" ]; then
    echo "Using flags $JOINED_FLAGS, from command line."
else
    echo "Using no additional flags."
fi


python run_glue.py /fly/task_configs/glue_peft_configs/cola.json --do_tune=True $FLAGS --project="monarch_glue_tune"  
python run_glue.py /fly/task_configs/glue_peft_configs/mrpc.json --do_tune=True $FLAGS --project="monarch_glue_tune"  
python run_glue.py /fly/task_configs/glue_peft_configs/qnli.json --do_tune=True $FLAGS --project="monarch_glue_tune"  
python run_glue.py /fly/task_configs/glue_peft_configs/rte.json  --do_tune=True $FLAGS --project="monarch_glue_tune"  
python run_glue.py /fly/task_configs/glue_peft_configs/sst-2.json --do_tune=True $FLAGS --project="monarch_glue_tune" 
python run_glue.py /fly/task_configs/glue_peft_configs/stsb.json --do_tune=True $FLAGS --project="monarch_glue_tune"  
python run_glue.py /fly/task_configs/glue_peft_configs/qqp.json  --do_tune=True $FLAGS --project="monarch_glue_tune"  
python run_glue.py /fly/task_configs/glue_peft_configs/mnli.json --do_tune=True $FLAGS --project="monarch_glue_tune"  
python run_glue.py /fly/task_configs/glue_peft_configs/wnli.json --do_tune=True $FLAGS --project="monarch_glue_tune" 

find . -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in $out_path to glue_submit.zip. Ready for submission."