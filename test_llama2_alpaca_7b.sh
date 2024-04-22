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
    --eval_accumulation_steps 1 \
    --use_auth \
    --output_dir  /fly/results/llama_mmlu/train \
    --do_eval \
    --bf16 \
    --dataset alpaca \
    --source_max_len 16 \
    --target_max_len 512 \
    --seed 0 \
    --hf_token=$HF_TOKEN "${FLAGS[@]}" \
    --do_mmlu_eval \
    --mmlu_split test 
