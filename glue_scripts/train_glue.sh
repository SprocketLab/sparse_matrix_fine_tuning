# !/bin/bash
# allow additional flags 
if [ "$1" != "" ]; then
    FLAGS="$1"
    echo "Using flags $FLAGS from command line."
else
    FLAGS=""
    echo "Using no additional flags."
fi
CUDA_VISIBLE_DEVICES=0 python run_glue.py /fly/task_configs/glue_peft_configs/cola.json   $FLAGS &
CUDA_VISIBLE_DEVICES=1 python run_glue.py /fly/task_configs/glue_peft_configs/mrpc.json   $FLAGS &
CUDA_VISIBLE_DEVICES=2 python run_glue.py /fly/task_configs/glue_peft_configs/qnli.json   $FLAGS &
CUDA_VISIBLE_DEVICES=3 python run_glue.py /fly/task_configs/glue_peft_configs/rte.json    $FLAGS &
CUDA_VISIBLE_DEVICES=4 python run_glue.py /fly/task_configs/glue_peft_configs/sst-2.json  $FLAGS &
CUDA_VISIBLE_DEVICES=5 python run_glue.py /fly/task_configs/glue_peft_configs/stsb.json   $FLAGS &
CUDA_VISIBLE_DEVICES=6 python run_glue.py /fly/task_configs/glue_peft_configs/qqp.json    $FLAGS &
CUDA_VISIBLE_DEVICES=7 python run_glue.py /fly/task_configs/glue_peft_configs/mnli.json   $FLAGS &
CUDA_VISIBLE_DEVICES=0  python run_glue.py /fly/task_configs/glue_peft_configs/wnli.json  $FLAGS
CUDA_VISIBLE_DEVICES=1 python run_glue.py task_configs/glue_peft_configs/stsb.json $FLAGS

$out_path = "results/monarch_roberta_glue"
# find all .tsv files in results/monarch_roberta_glue and zip them
find results/monarch_roberta_glue -maxdepth 2 -name "*.tsv" | zip -j results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in $out_path to glue_submit.zip. Ready for submission."