#!/bin/bash

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0

random_port(){
    # Random port
    MASTER_PORT=$((30000 + RANDOM % (99999 - 30000 + 1)))
    echo "MASTER_PORT=$MASTER_PORT"
}

export_world_info() {
    # Set world info for deepspeed
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "CUDA_VISIBLE_DEVICES is not set"
        NUM_GPUS=$(nvidia-smi -L | wc -l)
        CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))
        echo "Use all GPUs"
        export "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    else
        NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
}

random_port
export_world_info

# Train
accelerate launch \
    --main_process_port $MASTER_PORT \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds.json \
    trainer.py \

# Evaluate
# python eval.py