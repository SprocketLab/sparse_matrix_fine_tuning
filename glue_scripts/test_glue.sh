# Check if an argument was provided
DEVICE_ID=0
if [ "$1" != "" ]; then
    DEVICE_ID="$1"
fi  
echo "Using device $DEVICE_ID."

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES="$DEVICE_ID"
python run_glue.py task_configs/monarch_roberta_glue/peft_con/cola.json --do_train=False --wandb=False --resume_from_checkpoint=True 
python run_glue.py task_configs/monarch_roberta_glue/peft_con/mrpc.json --do_train=False --wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/monarch_roberta_glue/peft_con/qnli.json --do_train=False --wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/monarch_roberta_glue/peft_con/rte.json  --do_train=False --wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/monarch_roberta_glue/peft_con/sst-2.json --do_train=False --wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/monarch_roberta_glue/peft_con/stsb.json  --do_train=False --wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/monarch_roberta_glue/peft_con/qqp.json   --do_train=False --wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/monarch_roberta_glue/peft_con/mnli.json  --do_train=False --wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/monarch_roberta_glue/peft_con/wnli.json  --do_train=False --wandb=False --resume_from_checkpoint=True

# find all .tsv files in results/monarch_roberta_glue and zip them
find results/monarch_roberta_glue -maxdepth 2 -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in results/monarch_roberta_glue to glue_submit.zip. Ready for submission."
