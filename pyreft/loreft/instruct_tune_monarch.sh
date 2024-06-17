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

python train.py -task ultrafeedback \
-data_dir dataset \
-model meta-llama/Llama-2-7b-hf \
-seed 42 -l "3;9;18;24" -r 4 -p f5+l5 -e 9 -lr 5e-4 \
--intervention_type nothing \
-gradient_accumulation_steps 32 \
-batch_size 1 \
-eval_batch_size 2 \
--test_split test \
--use_normalized_template \
--max_length 768 \
-wu 0.03 -dtype bfloat16 "${FLAGS[@]}"