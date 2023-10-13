Before running anything, set up env by running 
```
docker compose build; docker compose up 
docker attach peft 
```

## Repo guide
Search for @Wenxuan in the comments for my code or important functions.
Check src/models/layers/monarch_linear.py for a basic setup of Monarch.
Check src/models/layers/blockdiag_butterfly_multiply.py to fully understand Monarch matmul.
Check src/ops/blockdiag_butterfly_einsum.py to understand D2S projection
Full finetuning: 
```
python finetune_roberta_agnews.py 
```
(change peft_config inside each script for ranks, etc.)
LoRA PEFT: 
```
python lora_roberta_agnews.py 
```
Monarch PEFT: 
``` 
python monarch_roberta_agnews.py 
```
modeling_roberta.py contains my modified Roberta supporting LoRA and Monarch.