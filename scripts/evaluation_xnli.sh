#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1

random_port(){
    # Random port
    MASTER_PORT=$((30000 + RANDOM % (99999-30000+1)))
    echo "MASTER_PORT=$MASTER_PORT"
}

check_4090() {
    # Check if the GPU is RTX 40 series
    if nvidia-smi | grep -q 'RTX 40'; then
        echo "RTX 40 series GPU detected, disabling NCCL P2P and IB"
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
    fi
}


export_world_info() {
    # Set world info for deepspeed
    # ref: https://github.com/microsoft/DeepSpeed/issues/1331
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "CUDA_VISIBLE_DEVICES is not set"
        NUM_GPUS=$(nvidia-smi -L | wc -l)
        # generate GPUS from index 0
        CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))
        echo "Use all GPUs"
        export "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        WID=`echo  {\"localhost\": [$CUDA_VISIBLE_DEVICES]} | base64`
    else
        # count CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        WID=`echo  {\"localhost\": [$CUDA_VISIBLE_DEVICES]} | base64`
    fi
}
random_port
check_4090
export_world_info
source activate multi-lingual

python -u -m deepspeed.launcher.launch --world_info=$WID \
    --master_port $MASTER_PORT \
    -- run_evaluation.py \
    --deepspeed \
    --task xnli \
    --llm_path LLaMAX/LLaMAX2-7B-XNLI --mt_path google/mt5-xl \
    --eval_batch_size 8 \
    --structure Linear \
    --init_checkpoint /root/LayAlign/outputs/LayAlign-xnli-test1/epoch_2_augmentation/pytorch_model.bin \
    --augmentation False \

