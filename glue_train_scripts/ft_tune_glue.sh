#!/bin/bash

# Default group name
GROUP_NAME="FT"

# Check if an argument is provided
if [ "$1" != "" ]; then
    GROUP_NAME="$1"
    echo "Using group name $GROUP_NAME from command line."
else
    echo "Using default group name $GROUP_NAME."


# Use the group name in your commands
python run_glue.py /fly/task_configs/cola.json --do_tune=True --group="$GROUP_NAME" --monarch=False 
python run_glue.py /fly/task_configs/mrpc.json --do_tune=True --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/qnli.json --do_tune=True --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/rte.json  --do_tune=True --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/sst-2.json --do_tune=True --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/stsb.json --do_tune=True --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/qqp.json  --do_tune=True --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/mnli.json --do_tune=True --group="$GROUP_NAME" --monarch=False
python run_glue.py /fly/task_configs/wnli.json --do_tune=True --group="$GROUP_NAME" --monarch=False
