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
python monarch_roberta.py --dataset ["cola", "ag-news] --peft
```
Use run_glue.py for other GLUE tasks (or cola)
```
export TASK_NAME=mrpc
python run_glue.py \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
``` 