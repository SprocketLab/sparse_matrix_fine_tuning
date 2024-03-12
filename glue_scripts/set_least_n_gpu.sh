# take one command line argument n (number of GPUs to use)
n=8
if [ "$1" != "" ]; then
    n="$1"
fi
echo "Using $n least memory GPUs."

CUDA_VISIBLE_DEVICES_get_n_least_memory_usage() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n \
        | paste -sd "," -)
    echo "CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')"
}
set_n_least_gpu() {
    echo "GPU Memory Usage:"
    local devices=$(CUDA_VISIBLE_DEVICES_get_n_least_memory_usage $1 | tail -1 | awk -F= '{ print $2 }')
    echo "set CUDA_VISIBLE_DEVICES=$devices"
    export CUDA_VISIBLE_DEVICES=$devices
}

set_n_least_gpu $n