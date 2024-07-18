PROJECT_NAME="monarch_GLUE_no_peft"


python run_glue.py /fly/task_configs/cola.json --do_tune=True --project="$PROJECT_NAME" --use_peft=False
python run_glue.py /fly/task_configs/mrpc.json --do_tune=True --project="$PROJECT_NAME" --use_peft=False
python run_glue.py /fly/task_configs/qnli.json --do_tune=True --project="$PROJECT_NAME" --use_peft=False
python run_glue.py /fly/task_configs/rte.json  --do_tune=True --project="$PROJECT_NAME" --use_peft=False
python run_glue.py /fly/task_configs/sst-2.json --do_tune=True --project="$PROJECT_NAME" --use_peft=False
python run_glue.py /fly/task_configs/stsb.json --do_tune=True --project="$PROJECT_NAME" --use_peft=False
python run_glue.py /fly/task_configs/qqp.json  --do_tune=True --project="$PROJECT_NAME" --use_peft=False
python run_glue.py /fly/task_configs/mnli.json --do_tune=True --project="$PROJECT_NAME" --use_peft=False
python run_glue.py /fly/task_configs/wnli.json --do_tune=True --project="$PROJECT_NAME" --use_peft=False
