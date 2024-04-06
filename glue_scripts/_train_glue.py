import subprocess
import sys
import os
import torch
from queue import deque
import argparse

def main():
    """
    Runs all GLUE tasks training,
    by specifying the IDs of GPUs you want to use in parallel.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="IDs of GPUs to use in parallel, separated by commas")
    args, extra_args = parser.parse_known_args()
    gpus = args.gpus.split(",")
    assert torch.cuda.device_count() >= len(gpus), f"Not enough GPUs for specifided IDs: {gpus}"
    
    os.chdir("/fly")
    tasks = deque(["cola", "sst-2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"])
    while tasks:
        command = []
        for gpu in gpus:
            if not tasks:
                break
            task = tasks.popleft()
            command += [f"CUDA_VISIBLE_DEVICES={gpu}", "python", "run_glue.py", f"task_configs/glue_peft_configs/{task}.json"] + extra_args
            
            # Run in parallel
            if gpu != gpus[-1]:
                command.append("&")
        subprocess.run(command)
        
if __name__ == "__main__":
    main()
    
    
