import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
from peft import LoraConfig
from LayAlign import LayAlign,LayAlignConfig
from huggingface_hub import login
import random
from types import SimpleNamespace

# torch.set_float32_matmul_precision("high")
# seed_everything(42)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu

set_seed(42)

# Custom Model using PyTorch Lightning
class MPTrainer(pl.LightningModule):
    def __init__(self, trainer_args):
        super(MPTrainer, self).__init__()
        
        self.tokenizer_m2m = AutoTokenizer.from_pretrained(trainer_args['encoder_model_name'])
        self.tokenizer_llm = AutoTokenizer.from_pretrained(trainer_args['llm_model_name'], use_fast=True)
        self.tokenizer_llm.pad_token = self.tokenizer_llm.eos_token
        self.tokenizer_llm.padding_side = "left"

        encoder_layers=trainer_args['encoder_layers']
        language_layers=trainer_args['language_layers']
        encoder_aligner_config = {
            "encoder_hidden_dim": trainer_args['encoder_hidden_dim'],
            "language_hidden_dim": trainer_args['language_hidden_dim'],
            "num_transformer_submodules": 1,
            "num_attention_heads": len(language_layers),
            "num_encoder_layers": len(encoder_layers),
            "num_language_layers": len(language_layers),
            "encoder_layers": encoder_layers,
            "language_layers": language_layers,
            "projector_type": "weighted_linear",
            "batch": trainer_args['batch_size'],
            "structure": "Linear"
        }
        
        encoder_aligner_config = SimpleNamespace(**encoder_aligner_config)
    
        model_config = LayAlignConfig(
            mt_path=trainer_args['encoder_model_name'],
            llm_path=trainer_args['llm_model_name'],
            max_gen_len=trainer_args['max_gen_len'],
            llm_bos_token_id=self.tokenizer_llm.bos_token_id,
            llm_pad_token_id=self.tokenizer_llm.pad_token_id,
            encoder_aligner_config=encoder_aligner_config,
            augmentation = True,
            lora_config = trainer_args['lora_config'],
            quantization_config=trainer_args['quantization_config'],
            num_embedding_tokens=trainer_args['num_embedding_tokens']
        )
    
        self.model = LayAlign(model_config)

        self.system_prompt = trainer_args['system_prompt']
        
        self.batch_size = trainer_args['batch_size']
        self.lr = trainer_args['lr']

        self.user_prompt_function = trainer_args['user_prompt_function']

        self.count_trainable_parameters()

    def forward(self, input_ids_prompt=None, mask_prompt=None, labels=None, mask_label=None, input_ids_m2m=None, attention_mask_m2m=None):
        output = self.model(
            input_ids_m2m, attention_mask_m2m,
            input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
            labels=labels, mask_label=mask_label
        )
        return output

    def training_step(self, batch, batch_idx):
        encoder_input, expected_output = batch
        
        encoded_inputs = self.tokenizer_m2m(
            encoder_input, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        
        input_ids_m2m = encoded_inputs["input_ids"]
        attention_mask_m2m = encoded_inputs["attention_mask"]

        llm_input_prompts = []
        
        user_prompts = [
            [{
                "role": "system", "content": self.system_prompt
            },
            {
                "role": "user", "content": self.user_prompt_function(encoder_input)
            }]
            for i in range(len(encoder_input))
        ]

        llm_input_prompts = [self.tokenizer_llm.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in user_prompts]
        
        llm_inputs = self.tokenizer_llm(
            llm_input_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        
        labels = [f"{sentence}{self.tokenizer_llm.eos_token}" for sentence in expected_output]
        
        llm_labels = self.tokenizer_llm(
            labels, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            add_special_tokens=False
        ).to(self.device)
        
        loss = self(
            input_ids_prompt=llm_inputs['input_ids'],
            mask_prompt=llm_inputs['attention_mask'],
            labels=llm_labels['input_ids'],
            mask_label=llm_labels['attention_mask'],
            input_ids_m2m=input_ids_m2m,
            attention_mask_m2m=attention_mask_m2m,
        )
        self.log("train_loss", loss, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        return loss

    def count_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Percentage: {round((trainable_params/total_params)*100,2)}%")

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = AdamW(params, lr=self.lr)
        return optimizer

    def save_weights(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")

    def load_weights(self, file_path: str, map_location=None):
        if map_location is None:
            map_location = self.device
        self.model.load_state_dict(torch.load(file_path, map_location=map_location))
        print(f"Model weights loaded from {file_path}")

    def evaluate_dataset(self, encoder_inputs, batch_size=1):
        response = []
        with torch.no_grad():
            for i in tqdm(range(0, len(encoder_inputs), batch_size)):
                batch_q = encoder_inputs[i:i + batch_size]
    
                    
                encoded_inputs = self.tokenizer_m2m(
                    batch_q, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                
                input_ids_m2m = encoded_inputs["input_ids"]
                attention_mask_m2m = encoded_inputs["attention_mask"]
        
                llm_input_prompts = []
                
                user_prompts = [
                    [{
                        "role": "system", "content": self.system_prompt
                    },
                    {
                        "role": "user", "content": self.user_prompt_function(batch_q)
                    }]
                    for i in range(len(batch_q))
                ]
        
                llm_input_prompts = [self.tokenizer_llm.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in user_prompts]
                
                llm_inputs = self.tokenizer_llm(
                    llm_input_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False
                ).to(self.device)
                
                generated_ids = self(
                    input_ids_prompt=llm_inputs['input_ids'],
                    mask_prompt=llm_inputs['attention_mask'],
                    input_ids_m2m=input_ids_m2m,
                    attention_mask_m2m=attention_mask_m2m,
                )
                
                # Decode responses
                batch_responses = self.tokenizer_llm.batch_decode(generated_ids, skip_special_tokens=True)
                response.extend(batch_responses)
                
                del encoded_inputs, llm_inputs
                torch.cuda.empty_cache()
            
        return response

# Dataset Class
class ExperimentDataset(Dataset):
    def __init__(self, encoder_inputs, expected_output):
        self.encoder_inputs = encoder_inputs
        self.expected_output = expected_output

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        return self.encoder_inputs[idx], self.expected_output[idx]

# token = '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'
token = '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'
login(token=token)

ds = load_dataset("Themira/en_si_news_classification_with_label_name")

train_ds = ds['train_en']

# Access the train and test splits
# train_ds = ds.filter(lambda example: example['lang'] in ['en', 'sw'])

print("Train split:", train_ds)

train_dataset = ExperimentDataset(
    train_ds['sentence'],
    train_ds['label']
)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

llm_model_name = "meta-llama/Llama-3.2-1B-Instruct"
encoder_model_name = "google/mt5-large"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4"
# )

quantization_config = None

encoder_layers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,]
language_layers = [i for i in range(16)]
encoder_hidden_dim = 1024
language_hidden_dim = 2048
num_embedding_tokens = -1
lr = 2e-5
epochs = 3
max_gen_len = 10
system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."

# lora_config = LoraConfig(
#     r=32,
#     lora_alpha=128,  # Maintains an effective scaling factor of 4
#     target_modules=["q_proj", "k_proj", "v_proj"],
#     lora_dropout=0.1,
#     bias="all",
#     task_type="CAUSAL_LM",
# )

lora_config = None

def user_prompt_function(enc_inputs):
    return f"### Instruction:\nClassify the given news sentence into one of the following categories.\nBusiness, Entertainment, Political, Sports, Science.\n\n### Response:"

train_args = {
    'llm_model_name': llm_model_name,
    'encoder_model_name': encoder_model_name,
    'system_prompt': system_prompt,
    'batch_size': batch_size,
    'lr': lr,
    'epochs': epochs,
    'user_prompt_function': user_prompt_function,
    'encoder_hidden_dim': encoder_hidden_dim,
    'language_hidden_dim': language_hidden_dim,
    'encoder_layers': encoder_layers,
    'language_layers': language_layers,
    'max_gen_len': max_gen_len,
    'quantization_config': quantization_config,
    'lora_config': lora_config,
    'num_embedding_tokens': num_embedding_tokens
}

model = MPTrainer(train_args)

print(model)

for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, requires_grad={param.requires_grad}, shape={param.shape}")

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='/kaggle/working/',  # Directory to save checkpoints
    filename='final_checkpoint',  # Fixed filename for the last checkpoint
    save_top_k=1,  # Save only the final checkpoint
    save_last=False,  # Ensure the last checkpoint is saved
    every_n_epochs=epochs  # Save only at the last epoch
)

trainer = Trainer(
    accelerator="gpu",
    max_epochs=epochs,
    precision="bf16",
    devices=1,
    callbacks=[checkpoint_callback],
)

print('training')
# Train the Model
trainer.fit(
    model,
    train_dataloaders=train_loader,
    # val_dataloaders=val_loader
    # ckpt_path="/root/LayAlign/final_checkpoint.ckpt"
)

# final_val_loss = trainer.callback_metrics.get("val_loss", None)
final_train_loss = trainer.callback_metrics.get("train_loss", None)
print(final_train_loss)

model.save_weights('/kaggle/working/model.pth')

model.eval()

model.to('cuda')

si_pred = model.evaluate_dataset(ds['test_si']['sentence'], batch_size=batch_size//2)
accuracy_si = accuracy_score(ds['test_si']['label'], si_pred)
f1_si = f1_score(ds['test_si']['label'], si_pred, average='weighted')  # Use 'binary', 'macro', or 'weighted' as needed

print(f"Accuracy Si: {accuracy_si}")
print(f"F1 Score Si: {f1_si}")

en_pred = model.evaluate_dataset(ds['test_en']['sentence'], batch_size=batch_size//2)
accuracy_en = accuracy_score(ds['test_en']['label'], en_pred)
f1_en = f1_score(ds['test_en']['label'], en_pred, average='weighted')  # Use 'binary', 'macro', or 'weighted' as needed

print(f"Accuracy En: {accuracy_en}")
print(f"F1 Score En: {f1_en}")
