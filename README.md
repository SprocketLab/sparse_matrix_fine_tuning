Before running anything, set up env by running 
```
docker compose build; docker compose up -d
docker attach peft 
python monarch_roberta.py --peft
```
Full training configs and logs will be stored in wandb. (You can also use your key)

## Repo guide
Search for @Wenxuan in the comments for my code or important functions.
Check src/models/layers/monarch_linear.py for a basic setup of Monarch.
Check src/models/layers/blockdiag_butterfly_multiply.py to fully understand Monarch matmul.
Check src/ops/blockdiag_butterfly_einsum.py to understand D2S projection
See src/models/modeling_roberta.py for my modified Roberta supporting LoRA and Monarch.

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
Use run_glue.py for GLUE tasks 
(Uses hyperparams from LoRA paper but reduces epochs slightly)
Training hyperparams are in train_configs/[task].json
Tune block size, LoRA/Monarch config in train_configs/peft_config.json.
```
python run_glue.py train_configs/cola.json
``` 