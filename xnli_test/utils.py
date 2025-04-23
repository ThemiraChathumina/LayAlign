import os
import random
import torch
from tqdm import tqdm
from read_data import llm_input_features
import numpy as np

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu

def save_with_accelerate(accelerator, model, output_dir, model_name='pytorch_model.bin'):
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir
    accelerator.wait_for_everyone()
    accelerator.save_model(model, output_file, max_shard_size="30GB",safe_serialization=False)


def get_train_ds_config(train_batch_size=1,
                        train_micro_batch_size_per_gpu=1,
                        lr=2e-5,
                        gradient_accumulation_steps=1,
                        offload=True,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        warm_step=0,
                        train_step=0):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "steps_per_print": 2000,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": False,
        },
        "gradient_clipping": 1.0,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [
                    0.8,
                    0.999
                ],
                "eps": 1e-8
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
            "total_num_steps": train_step,
            "warmup_num_steps": warm_step
            }
        },
    }

def evaluate_classification(model, test_set, tokenizer_llm, max_gen_len, use_prompt):
    model.eval()
    results_list = []
    hit = 0
    step_trange = tqdm(test_set)
    preds, golds = [], []
    for test_step in step_trange:
        prompts = test_step['prompt']
        targets = test_step['target']
        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            add_bos_token = False
            add_eos_token = False
            input_ids_prompt, mask_prompt = llm_input_features(prompts, tokenizer_llm, max_gen_len, add_bos_token,
                                                           add_eos_token)
        generate_ids = model(prompts,
                             input_ids_prompt=input_ids_prompt,
                             mask_prompt=mask_prompt)

        results = tokenizer_llm.batch_decode(generate_ids,
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)

        preds += results
        golds += targets

        for result, prompt, target in zip(results, prompts, targets):
            result = result.strip()
            results_list.append({
                'prompt': prompt,
                'prediction': result,
                'answer': target
            })
            if target == result:
                hit += 1

        acc = round(hit / len(results_list) * 100, 2)
        loss_show = 'Acc:' + str(acc)
        step_trange.set_postfix_str(loss_show)

    acc = round(hit / len(results_list) * 100, 2)
    return acc, results_list
