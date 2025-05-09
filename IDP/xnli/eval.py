import torch.fx
from transformers import AutoTokenizer
import torch
import argparse
import ast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import os
import sys
import deepspeed
from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import math
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import nn
from datasets import load_dataset
from torch.utils.data import Dataset
from huggingface_hub import login, upload_file

from utils import get_train_ds_config, set_seed, evaluate_classification
from model import MPTModel
from read_data import ExperimentDataset, llm_input_features

from Configs import XNLIConfigs

def construct_prompt(sample):
    return f"### Instruction:\nNews Sentence: {sample}\nClassify the given news sentence into one of the following categories.\nBusiness, Entertainment, Political, Sports, Science.\n\n### Response:"


def apply_chat_template(system_prompt, user_prompt, tokenizer_llm, tokenize=False, add_generation_prompt=True):
    # This function is a placeholder for the actual implementation of applying a chat template.
    # In a real scenario, this would format the prompt according to the requirements of the model.
    user_prompts = [{
                        "role": "system", "content": system_prompt
                    },
                    {
                        "role": "user", "content": user_prompt
                    }]
    chat_prompt = tokenizer_llm.apply_chat_template(user_prompts, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    return chat_prompt

def main():
    confings = XNLIConfigs()

    llm_path = "meta-llama/Llama-3.2-1B-Instruct"
    mt_path = "google/mt5-large"
    ext_path = "facebook/nllb-200-distilled-600M"
    max_seq_len = 512
    max_gen_len = 512
    eval_batch_size = 4
    augmentation = False
    save_name = "no_aug"
    task = "xnli"
        
    system_prompt = confings.system_prompt

    result_path_base = f'./results/{save_name}/{task}/'

    token = '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'
    login(token=token)

    languages = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    test_sets = {}
    for lang in languages:
        ds = load_dataset("facebook/xnli", lang)
        test_ds = ds['test']
        test_set = []
        for i in range(len(test_ds)):
            example = test_ds[i]
            premise = example['premise']
            hypothesis = example['hypothesis']
            labels = {0:'Entailment', 1:'Neutral', 2:'Contradiction'}
            sentence = f"{confings.getPremise()}: {premise.strip()} {confings.getHypothesis()}: {hypothesis.strip()} {confings.getLabel()}:"
            label = f"{labels[int(example['label'])]}"
            sample = {
                'prompt': sentence,
                'target': label
            }
            test_set.append(sample)
        test_sets[lang] = test_set

    os.makedirs(result_path_base, exist_ok=True)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    # tokenizer_llm.pad_token = "[PAD]"
    print(json.dumps({
        'llm_path': llm_path,
        'mt_path': mt_path,
        'ext_path': ext_path,
        'max_seq_len': max_seq_len,
        'max_gen_len': max_gen_len,
        'save_name': save_name,
        'result_path_base': result_path_base
    }, indent=2))
    print("cuda available: " , torch.cuda.is_available())
    train_micro_batch_size_per_gpu = 4
    train_batch_size = 4
    gpu_num = torch.cuda.device_count()
    gradient_accumulation = 1
    # assert train_micro_batch_size_per_gpu * gpu_num * gradient_accumulation == train_batch_size
    ds_config = get_train_ds_config(train_batch_size=train_batch_size,
                                    train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
                                    gradient_accumulation_steps=gradient_accumulation,
                                    )

    model_config = {
        'mt_path': mt_path,
        'ext_path': ext_path,
        'llm_path': llm_path,
        'max_gen_len': max_gen_len,
        'llm_bos_token_id': tokenizer_llm.bos_token_id,
        'llm_pad_token_id': tokenizer_llm.pad_token_id,
        'augmentation' :  augmentation,
        'max_seq_len': max_seq_len
    }
    
    init_checkpoint = confings.checkpoint

    

    model = MPTModel(model_config)
    if init_checkpoint is not None:
        init_checkpoint = init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        #model_dict = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint, True)
        print('mapping init from:', init_checkpoint)
    # model.to('cuda')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, __ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=parameters,
        training_data=None)
    scores_map = {}
    avg = 0
    for test_lang in test_sets:
        test_set = test_sets[test_lang]
        test_sampler = SequentialSampler(test_set)
        test_set = ExperimentDataset(test_set)
        test_set = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=eval_batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=1,
            drop_last=False)
        acc, results_list = evaluate_classification(model, test_set, tokenizer_llm, max_gen_len, augmentation, system_prompt)
        
        print('test_lang:', test_lang, 'acc:', acc)
        scores_map[test_lang] = acc
        result_path = f'{result_path_base}/{test_lang}.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
        avg += acc
    print(scores_map)
    print('Average accuracy :', round(avg / len(test_sets), 1))
    score_path = f'{result_path_base}/scores.tsv'
    with open(score_path, 'w', encoding='utf-8') as f:
        for lang in scores_map:
            score = scores_map[lang]
            f.write(f'{lang}\t{score}\n')



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(0)

    main()