#!/bin/bash

# Default group name
GROUP_NAME="FT"

# Check if an argument is provided
if [ "$1" != "" ]; then
    GROUP_NAME="$1"
    echo "Using group name $GROUP_NAME from command line."
else
    echo "Using default group name $GROUP_NAME."
fi

# Use the group name in your commands
python run_glue.py /fly/task_configs/cola.json  --group="$GROUP_NAME" --monarch=False 
python run_glue.py /fly/task_configs/mrpc.json  --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/qnli.json  --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/rte.json   --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/sst-2.json  --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/stsb.json  --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/qqp.json   --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/mnli.json  --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/wnli.json  --group="$GROUP_NAME" --monarch=False

find . -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in $out_path to glue_submit.zip. Ready for submission."