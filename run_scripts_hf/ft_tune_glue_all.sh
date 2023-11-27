# !/bin/bash

# Naive way to implement multiprocess training...
# Should use python mp spawn for gpu selection, job queueing, etc.
python run_glue.py /fly/task_configs/cola.json --do_tune=True --group="FT" --monarch=False 
python run_glue.py /fly/task_configs/mrpc.json --do_tune=True --group="FT" --monarch=False
python run_glue.py /fly/task_configs/qnli.json --do_tune=True --group="FT" --monarch=False
python run_glue.py /fly/task_configs/rte.json  --do_tune=True --group="FT" --monarch=False
python run_glue.py /fly/task_configs/sst-2.json --do_tune=True --group="FT" --monarch=False
python run_glue.py /fly/task_configs/stsb.json --do_tune=True --group="FT" --monarch=False
python run_glue.py /fly/task_configs/qqp.json  --do_tune=True --group="FT" --monarch=False
python run_glue.py /fly/task_configs/mnli.json --do_tune=True --group="FT" --monarch=False
python run_glue.py /fly/task_configs/wnli.json --do_tune=True --group="FT" --monarch=False

# $out_path = "results/monarch_roberta_glue"
# # find all .tsv files in results/monarch_roberta_glue and zip them
# find . -name "*.tsv" | zip $out_path/glue_submit.zip -@
# echo "Zipped all .tsv files in $out_path to glue_submit.zip. Ready for submission."