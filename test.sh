CUDA_VISIBLE_DEVICES=1 python run_glue.py /fly/task_configs/glue_peft_configs/cola.json --project=monarch_glue_tune --group="dense rank 4" --blk_r=1 --use_wandb=False
echo "Switched to background!"