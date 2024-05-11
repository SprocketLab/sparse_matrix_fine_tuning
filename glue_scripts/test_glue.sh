# Check if an argument was provided
DEVICE_ID=0
if [ "$1" != "" ]; then
    DEVICE_ID="$1"
fi  
echo "Using device $DEVICE_ID."

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES="$DEVICE_ID"
python run_glue.py task_configs/roberta_glue/cola.json --do_train=False --use_wandb=False --resume_from_checkpoint=True 
python run_glue.py task_configs/roberta_glue/mrpc.json --do_train=False --use_wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/roberta_glue/qnli.json --do_train=False --use_wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/roberta_glue/rte.json  --do_train=False --use_wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/roberta_glue/sst-2.json --do_train=False --use_wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/roberta_glue/stsb.json  --do_train=False --use_wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/roberta_glue/qqp.json   --do_train=False --use_wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/roberta_glue/mnli.json  --do_train=False --use_wandb=False --resume_from_checkpoint=True
python run_glue.py task_configs/roberta_glue/wnli.json  --do_train=False --use_wandb=False --resume_from_checkpoint=True

# find all .tsv files in results/monarch_roberta_glue and zip them
find results/monarch_roberta_glue -maxdepth 2 -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in results/monarch_roberta_glue to glue_submit.zip. Ready for submission."
