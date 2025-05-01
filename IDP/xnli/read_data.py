import math
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
import json
from datasets import load_dataset
from torch.utils.data import Dataset
from huggingface_hub import login, upload_file

class ExperimentDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample

def read_dataset(path):
    if 'jsonl' in path:
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(json.loads(line))
    elif 'json' in path:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if isinstance(dataset, dict):
            if 'data' in dataset:
                dataset = dataset['data']
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = f.readlines()
    return dataset

def llm_input_features(input_texts_llm, tokenizer_llm,
                         max_seq_len, add_bos_token, add_eos_token):
    tokenizer_llm.add_bos_token = add_bos_token
    # tokenizer_llm.add_eos_token = add_eos_token
    if add_eos_token:
        input_texts_llm = [f"{prompt}{tokenizer_llm.eos_token}" for prompt in input_texts_llm]
    encoding_llm = tokenizer_llm(input_texts_llm,
                         padding='longest',
                         max_length=max_seq_len,
                         truncation=True,
                         add_special_tokens = False,
                         return_tensors="pt")
    input_ids_llm = encoding_llm.input_ids.cuda()
    attention_mask_llm = encoding_llm.attention_mask.cuda()
    attention_mask_llm[:,-1] = 1.0
    return input_ids_llm, attention_mask_llm

def read_dataset(path):
    if 'jsonl' in path:
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(json.loads(line))
    elif 'json' in path:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if isinstance(dataset, dict):
            if 'data' in dataset:
                dataset = dataset['data']
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = f.readlines()
    return dataset