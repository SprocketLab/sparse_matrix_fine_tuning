# Check if an argument was provided
DEVICE_ID=0
if [ "$1" != "" ]; then
    DEVICE_ID="$1"
fi  
echo "Using device $DEVICE_ID."

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES="$DEVICE_ID"
python run_glue.py task_configs/cola.json --do_train=False --use_wandb=False 
python run_glue.py task_configs/mrpc.json --do_train=False --use_wandb=False
python run_glue.py task_configs/qnli.json --do_train=False --use_wandb=False
python run_glue.py task_configs/rte.json  --do_train=False --use_wandb=False
python run_glue.py task_configs/sst-2.json --do_train=False --use_wandb=False
python run_glue.py task_configs/stsb.json  --do_train=False --use_wandb=False
python run_glue.py task_configs/qqp.json   --do_train=False --use_wandb=False
python run_glue.py task_configs/mnli.json  --do_train=False --use_wandb=False
python run_glue.py task_configs/wnli.json  --do_train=False --use_wandb=False

# find all .tsv files in results/monarch_roberta_glue and zip them
find . -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in results/monarch_roberta_glue to glue_submit.zip. Ready for submission."