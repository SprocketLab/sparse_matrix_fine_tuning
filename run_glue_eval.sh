# Check if an argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <device_id>"
    exit 1
fi

# Set the CUDA_VISIBLE_DEVICES environment variable
CUDA_VISIBLE_DEVICES="$1" python run_glue.py train_configs/cola.json --do_train=False
CUDA_VISIBLE_DEVICES="$1" python run_glue.py train_configs/mrpc.json --do_train=False
CUDA_VISIBLE_DEVICES="$1" python run_glue.py train_configs/qnli.json --do_train=False
CUDA_VISIBLE_DEVICES="$1" python run_glue.py train_configs/rte.json  --do_train=False
CUDA_VISIBLE_DEVICES="$1" python run_glue.py train_configs/sst-2.json --do_train=False
CUDA_VISIBLE_DEVICES="$1" python run_glue.py train_configs/stsb.json  --do_train=False
CUDA_VISIBLE_DEVICES="$1" python run_glue.py train_configs/qqp.json   --do_train=False
CUDA_VISIBLE_DEVICES="$1" python run_glue.py train_configs/mnli.json  --do_train=False
CUDA_VISIBLE_DEVICES="$1" python run_glue.py train_configs/wnli.json  --do_train=False

# find all .tsv files in results/monarch_roberta_glue and zip them
find . -name "*.tsv" | zip results/monarch_roberta_glue/glue_submit.zip -@
echo "Zipped all .tsv files in results/monarch_roberta_glue to glue_submit.zip. Ready for submission."