Before running anything, set up env by running (the docker-compose.yml uses my WANDB_API_KEY by default)
```
docker compose build; docker compose up -d
hostname > hostname.txt # to not confuse machines in wandb
docker attach peft
```
Full training configs and logs will be stored in wandb. (You can use your key or flag --use_wandb=False)

## Repo guide
Search for @Wenxuan in the comments for my code or important functions.\

Check src/models/layers/blockdiag_butterfly_multiply.py to ***fully understand Monarch matmul.***\
Check src/models/layers/monarch_linear.py for a basic setup of Monarch.\
Check src/ops/blockdiag_butterfly_einsum.py to understand D2S projection.\
See src/models/modeling_roberta.py for my modified Roberta supporting LoRA and Monarch.

Full finetuning: 
```
python finetune_roberta.py 
```
LoRA PEFT (modify peft_config for ranks, etc.): 
```
python lora_roberta.py 
```
Monarch PEFT: 
``` 
python monarch_roberta.py  --peft
```
For GLUE tasks 
Training hyperparams are in task_configs/[task].json
Tune block size, rank, etc. in task_configs/peft_config.json.
Or try run_glue_hf.py with default args below
```
export WANDB_API_KEY="YOUR_KEY"
python run_glue.py task_configs/glue_peft_configs/cola.json --do_tune=False --use_wandb=True 
```
