# Initialize an array for flags
FLAGS=()

# Loop through all the arguments
for arg in "$@"
do
    # Check if the argument is in --key=value format
    if [[ $arg == --* ]]; then
        # Append the argument to FLAGS array, preserving the quoting
        FLAGS+=("$arg")
    fi
done

# Check if FLAGS array is not empty
if [ ${#FLAGS[@]} -ne 0 ]; then
    echo "Using flags ${FLAGS[@]} from the command line."
else
    echo "Using no additional flags."
fi


python qlora_monarch.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --eval_accumulation_steps 2 \
    --output_dir  /fly/results/llama_mmlu \
    --logging_steps 40 \
    --save_strategy steps \
    --save_steps 187 \
    --save_total_limit 10 \
    --evaluation_strategy steps \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --bf16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset alpaca \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --seed 1 \
    --hf_token=$HF_TOKEN "${FLAGS[@]}" \
    --do_mmlu_eval \
    --mmlu_split test \
    --load_best_model_at_end \
    --max_steps 1875 \
    # --num_train_epochs 2  #
