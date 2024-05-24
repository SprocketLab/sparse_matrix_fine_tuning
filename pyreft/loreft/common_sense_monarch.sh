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

python train.py -task commonsense \
-data_dir dataset \
-model yahma/llama-7b-hf \
-seed 42 \
-l all -r 8 -p f7+l7 -e 3 -lr 5e-4 \
--intervention_type nothing \
-gradient_accumulation_steps 16 \
-batch_size 2 \
-eval_batch_size 4 \
--dropout 0.00 \
--test_split test \
--use_normalized_template \
--share_weights \
--warmup_ratio 0.1 \
--greedy_decoding \
"${FLAGS[@]}"