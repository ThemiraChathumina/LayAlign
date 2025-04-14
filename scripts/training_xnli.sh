#!/bin/bash

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

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
source activate multi-lingual

# Set batch sizes
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=4   # 2 GPUs × 2 batch size each
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))

# Run only the augmentation stage
accelerate launch \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/zero2_dp.json \
    finetune.py \
    --llm_path LLaMAX/LLaMAX2-7B \
    --mt_path google/mt5-xl \
    --stage_name augmentation \
    --task xnli \
    --train_num 30000 \
    --train_batch_size $TOTAL_BATCH_SIZE \
    --train_micro_batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --gradient_accumulation $GRADIENT_ACC_STEPS \
    --augmentation True \
    --epoch_num 3 \
    --max_seq_len 200 \
    --max_gen_len 200 \
    --eval_batch_size $BATCH_SIZE_PER_GPU \
    --dev_size 3000 \
    --logging_steps 10 \
    --lr 3e-5 \
    --save_name LayAlign-xnli-test1 \
    --warm_rate 0.05 \
    --structure Linear \
    --lr_scheduler_name cosine
