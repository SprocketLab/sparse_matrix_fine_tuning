# MoRe Fine-Tuning with 10x Fewer Parameters
Official repository for ICML 2024 paper "MoRe Fine-Tuning with 10x Fewer Parameters". Using hardware efficient block-diagonal matrices, we surpass LoRA's performance with 10x fewer parameters on average, very little rank tuning and no alpha scaler. Our approach is also more memory and runtime efficient on standard reasoning tasks, with **our Llama 7B beating LoRA-trained Llama 13B on Commonsense reasoning**.

Paper: https://arxiv.org/abs/2408.17383

## News
- [2024/06] Our paper is accepted by two ICML workshops: ES-FOMO and FM-wild!

## MoRe Implementation
* `src/models/layers/monarch_linear.py` contains the code for MoRe adapter.
* `src/models/layers/blockdiag_butterfly_multiply.py` contains the code for block-diagonal monarch matmul.
## Setup
We highly recommend using docker for stable environment building, but any decent PyTorch + Huggingface environment should work.
```
docker compose build; docker compose up -d
hostname > hostname.txt # to not confuse machines in wandb
docker attach peft
```

## Usage
### To reproduce results
* GLUE tasks: `python run_glue.py /fly/task_configs/monarch_roberta_glue/cola.json`.\
* For reasoning tasks, first load datasets using `bash pyreft/loreft/load_datasets.sh`.
* Math reasoning: `CUDA_VISIBLE_DEVICES=0 bash  pyreft/loreft/math_monarch.sh`.
* Commonsense reasoning: `CUDA_VISIBLE_DEVICES=0 bash pyreft/loreft/common_sense_monarch.sh`.

You can manually modify the hyperparameters in `task_configs/llama` for reasoning tasks and
`task_configs/monarch_roberta_glue` for GLUE tasks.
