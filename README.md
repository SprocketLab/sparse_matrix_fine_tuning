Before running anything, set up env by running 
```
docker compose build; docker compose up -d
hostname > hostname.txt # to not confuse machines in wandb
docker attach peft
```
Full training configs and logs will be stored in wandb. (You can use your key or flag --use_wandb=False)

## Repo guide
Search for @Wenxuan in the comments for my code or important functions.\

Check `fly_src/models/layers/blockdiag_butterfly_multiply.py` to ***understand Monarch matmul.***\
Check `fly_src/models/layers/monarch_linear.py` for a basic setup of Monarch.\
Check `fly_src/ops/blockdiag_butterfly_einsum`.py to understand D2S projection.\
See `fly_src/models/modeling_roberta.py` for my modified Roberta supporting Monarch adapter.

Full finetuning: 
```
python finetune_roberta.py 
```
(change peft_config inside each script for ranks, etc.)
LoRA PEFT: 
```
python lora_roberta.py 
```
Monarch PEFT: 
``` 
python monarch_roberta.py  --peft
```
For GLUE tasks 
Training hyperparams are in task_configs/[task].json
Tune block size, rank, etc. in task_configs/peft_monarch.json.
Or try run_glue_hf.py with default args below
```
export WANDB_API_KEY="YOUR_KEY"
python run_glue.py task_configs/glue_peft_configs/cola.json --do_tune=False --use_wandb=True 
```
