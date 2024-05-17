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

python train.py -task glue \
-train_dataset cola \
-model FacebookAI/roberta-base \
-seed 42 \
-l all \
-r 1 \
-p f1 \
-e 12 \
-lr 3e-4 \
-type MoReIntervention \
-gradient_accumulation_steps 1 \
-batch_size 32 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--metric_for_best_model matthews_correlation \
--dropout 0.05 \
--weight_decay 0.00000 \
--warmup_ratio 0.09 \
--logging_steps 20 \
--allow_cls_grad "${FLAGS[@]}"