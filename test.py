from types import SimpleNamespace
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import torch
import wandb
import torch.nn.functional as F
import math
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, get_scheduler
# from liger_kernel.transformers import AutoLigerKernelForCausalLM
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
import sys
from layer_wise_aligner import EncoderAligner
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig, AdaptionPromptConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from lightning.pytorch.strategies import DeepSpeedStrategy
from huggingface_hub import login

torch.set_float32_matmul_precision("high")
seed_everything(42)
root_path = "/root/multilingual-p-tuning/"

class Mapper(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mapper, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MultiheadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads = 8):
        """
        Initializes the multi-head cross attention layer with layer normalization.
        :param embed_dim: int, embedding dimension (should be divisible by num_heads)
        :param num_heads: int, number of attention heads
        """
        super(MultiheadCrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection layers: input -> queries for encoder 1;
        # keys and values from encoder 2.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final output projection.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization after attention.
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, enc1_embedding, enc1_attention_mask, enc2_embedding, enc2_attention_mask):
        """
        Forward pass for multi-head cross attention.
        :param enc1_embedding: Tensor of shape (batch_size, num_tokens_1, embed_dim) -> Queries
        :param enc1_attention_mask: Tensor of shape (batch_size, num_tokens_1)
        :param enc2_embedding: Tensor of shape (batch_size, num_tokens_2, embed_dim) -> Keys and Values
        :param enc2_attention_mask: Tensor of shape (batch_size, num_tokens_2)
        :return: Tensor of shape (batch_size, num_tokens_1, embed_dim)
        """
        batch_size = enc1_embedding.shape[0]
        
        # 1. Project the embeddings to queries, keys, and values.
        Q = self.q_proj(enc1_embedding)  # (B, num_tokens_1, embed_dim)
        K = self.k_proj(enc2_embedding)  # (B, num_tokens_2, embed_dim)
        V = self.v_proj(enc2_embedding)  # (B, num_tokens_2, embed_dim)
        
        # 2. Reshape and transpose for multi-head attention.
        # New shape: (B, num_tokens, num_heads, head_dim) then transpose to (B, num_heads, num_tokens, head_dim)
        def reshape_to_heads(x, num_heads, head_dim):
            B, T, _ = x.size()
            x = x.view(B, T, num_heads, head_dim)
            return x.transpose(1, 2)  # (B, num_heads, T, head_dim)
        
        Q = reshape_to_heads(Q, self.num_heads, self.head_dim)  # (B, num_heads, num_tokens_1, head_dim)
        K = reshape_to_heads(K, self.num_heads, self.head_dim)  # (B, num_heads, num_tokens_2, head_dim)
        V = reshape_to_heads(V, self.num_heads, self.head_dim)  # (B, num_heads, num_tokens_2, head_dim)
        
        # 3. Compute the scaled dot-product attention scores.
        # Scores shape: (B, num_heads, num_tokens_1, num_tokens_2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. Apply the key attention mask (from encoder 2).
        # Expand mask from (B, num_tokens_2) to (B, 1, 1, num_tokens_2)
        if enc2_attention_mask is not None:
            mask = enc2_attention_mask.unsqueeze(1).unsqueeze(2)  # broadcastable mask
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Compute the attention weights.
        attention_weights = F.softmax(scores, dim=-1)
        
        # 6. Compute the weighted sum of values.
        # (B, num_heads, num_tokens_1, head_dim)
        out_heads = torch.matmul(attention_weights, V)
        
        # 7. Combine heads:
        # Transpose back to (B, num_tokens_1, num_heads, head_dim) then flatten.
        out_heads = out_heads.transpose(1, 2).contiguous()  # (B, num_tokens_1, num_heads, head_dim)
        output = out_heads.view(batch_size, -1, self.embed_dim)  # (B, num_tokens_1, embed_dim)
        
        # 8. Project the concatenated output.
        output = self.out_proj(output)
        
        # 9. (Optional) Apply the encoder 1 mask to zero out outputs for padded tokens.
        if enc1_attention_mask is not None:
            output = output * enc1_attention_mask.unsqueeze(-1)
        
        # 10. Apply layer normalization.
        output = self.layer_norm(output)
        
        return output


class FusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=1):
        super(FusionBlock, self).__init__()
        self.layers = nn.ModuleList(
            [MultiheadCrossAttention(embed_dim, num_heads) for _ in range(num_layers)]
        )
        
    def forward(self, enc1_embedding, enc1_attention_mask, enc2_embedding, enc2_attention_mask):
        out = enc1_embedding
        for layer in self.layers:
            out = layer(out, enc1_attention_mask, enc2_embedding, enc2_attention_mask)
        return out


class MultilingualEmbeddingModel(nn.Module):
    ALLOWED_MODELS = {
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "facebook/nllb-200-distilled-600M",
        "facebook/nllb-200-1.3B",
        "google/mt5-small",
        "google/mt5-base",
        "google/mt5-large",
        "DKYoon/mt5-small-lm-adapt",
        "DKYoon/mt5-large-lm-adapt",
        "DKYoon/mt5-xl-lm-adapt",
        "facebook/nllb-200-distilled-1.3B"
    }
    
    def __init__(self, embedding_model_base, embedding_model_ext, num_embedding_tokens = -1, 
                 freeze_embedding = True):
        super().__init__()

        if embedding_model_base not in self.ALLOWED_MODELS or embedding_model_ext not in self.ALLOWED_MODELS:
            raise ValueError(f"Model is not in allowed models: {self.ALLOWED_MODELS}")
        
        self.embedding_model_base = AutoModel.from_pretrained(embedding_model_base)
        if "nllb" in embedding_model_base or "mt5" in embedding_model_base:
            self.embedding_model_base = self.embedding_model_base.encoder 

        # self.embedding_model_ext = AutoModel.from_pretrained(embedding_model_ext)
        # if "nllb" in embedding_model_ext or "mt5" in embedding_model_ext:
        #     self.embedding_model_ext = self.embedding_model_ext.encoder 
            
        self.tokenizer_base = AutoTokenizer.from_pretrained(embedding_model_base)
        # self.tokenizer_ext = AutoTokenizer.from_pretrained(embedding_model_ext)
        
        self.num_embedding_tokens = num_embedding_tokens

        self.embedding_dim_base = self.embedding_model_base.config.hidden_size
        # self.embedding_dim_ext = self.embedding_model_ext.config.hidden_size
        self.embedding_dim = self.embedding_dim_base

        # self.mapper_base = Mapper(self.embedding_dim_base, self.embedding_dim)
        # self.mapper_ext = Mapper(self.embedding_dim_ext, self.embedding_dim)

        # self.fusion_block = FusionBlock(self.embedding_dim)

        self.freeze_embedding = freeze_embedding
        if freeze_embedding:
            for param in self.embedding_model_base.parameters():
                param.requires_grad = False
            # for param in self.embedding_model_ext.parameters():
            #     param.requires_grad = False

        self.learnable_queries_base = None
        self.learnable_queries_ext = None
        
        # If using prepended queries, initialize them.
        if num_embedding_tokens > -1:
            self.learnable_queries_base = nn.Parameter(torch.randn(num_embedding_tokens, self.embedding_dim_base))
            # self.learnable_queries_ext = nn.Parameter(torch.randn(num_embedding_tokens, self.embedding_dim_ext))

    def get_input_embeddings(self, model, input_ids):
        if "M2M" in model.__class__.__name__:
            return model.embed_tokens(input_ids)
        return model.get_input_embeddings()(input_ids)
    
    def get_last_hidden_states(self, encoded_inputs, model, tokenizer, queries = None):
        tokenized = tokenizer(
            encoded_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        input_ids = tokenized["input_ids"].to(next(self.parameters()).device)
        attention_mask = tokenized["attention_mask"].to(next(self.parameters()).device)
        
        batch_size = input_ids.shape[0]
        
        if self.num_embedding_tokens > -1:
            inputs_embeds = self.get_input_embeddings(model, input_ids)  # [B, L, D]
            
            queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Q, D]
            
            combined_inputs = torch.cat([queries, inputs_embeds], dim=1)  # [B, Q+L, D]
            
            query_mask = torch.ones(batch_size, self.num_embedding_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
            combined_attention_mask = torch.cat([query_mask, attention_mask], dim=1)  # [B, Q+L]
            
            outputs = model(inputs_embeds=combined_inputs, attention_mask=combined_attention_mask, outputs_hidden_states=True)
            
            return outputs, combined_attention_mask
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, outputs_hidden_states=True)
            return outputs, attention_mask
    
    def forward(self, encoded_inputs):
        base_embeddings, base_attention_mask = self.get_last_hidden_states(
            encoded_inputs, 
            self.embedding_model_base, 
            self.tokenizer_base,
            self.learnable_queries_base
        )

        # ext_embeddings, ext_attention_mask = self.get_last_hidden_states(
        #     encoded_inputs, 
        #     self.embedding_model_ext, 
        #     self.tokenizer_ext,
        #     self.learnable_queries_ext
        # )

        # base_embeddings = self.mapper_base(base_embeddings)
        # ext_embeddings = self.mapper_ext(ext_embeddings)

        # fused_embeddings = self.fusion_block(base_embeddings, base_attention_mask, ext_embeddings, ext_attention_mask)

        return base_embeddings, base_attention_mask
    

class MPTModel(nn.Module):
    def __init__(self, llm_model_name, embedding_model_base, embedding_model_ext, num_embedding_tokens, llm_bos_token_id, llm_pad_token_id, freeze_embedding=True, quantization_config=None, lora_config=None):
        super(MPTModel, self).__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, 
            quantization_config=quantization_config,
            # attn_implementation="flash_attention_2"
        )
        self.base_model_embedding_layer = self.base_model.get_input_embeddings()
        self.base_embedding_size = self.base_model.config.hidden_size
        self.llm_bos_token_id = llm_bos_token_id
        self.llm_pad_token_id = llm_pad_token_id
        if self.llm_bos_token_id is None:
            self.llm_bos_token_id = self.llm_pad_token_id
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.embeddimg_model = MultilingualEmbeddingModel(
            embedding_model_base,
            embedding_model_ext,
            num_embedding_tokens,
            freeze_embedding
        )
        
        encoder_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        language_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        self.encoder_aligner_config = {
            "encoder_hidden_dim": 2048,
            "language_hidden_dim": 4096,
            "num_transformer_submodules": 1,
            "num_attention_heads": 32,
            "num_encoder_layers": len(encoder_layers),
            "num_language_layers": len(language_layers),
            "encoder_layers": encoder_layers,
            "language_layers": language_layers,
            "projector_type": "weighted_linear",
            "batch": 2,
            "structure": "Linear"
        }
        
        self.encoder_aligner_config = SimpleNamespace(**self.encoder_aligner_config)

        self.mapper = Mapper(self.embeddimg_model.embedding_dim, self.base_embedding_size)
        self.encoder_aligner = EncoderAligner(self.encoder_aligner_config)
        
        peft_config = AdaptionPromptConfig(
            adapter_layers = self.encoder_aligner_config.language_layers
        )
        
        self.base_model = get_peft_model(self.base_model, peft_config).base_model
        
        if lora_config:
            self.lora_config = lora_config
            self.base_model = get_peft_model(self.base_model, self.lora_config)

        for name, param in self.base_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True  

    def squeeze_pad(self, hidden_states, masks):
        x_01 = (masks != 0).long()

        seq_num_len = x_01.size(1)
        offset = torch.tensor([(i + 1) for i in range(seq_num_len)], dtype=torch.long).to(x_01.device)
        offset = offset.unsqueeze(dim=0).expand_as(x_01)
        x_01 *= offset
        _, idx = x_01.sort(1, descending=False)

        masks = masks.gather(1, idx)
        idx = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states.gather(1, idx)

        bs, seq_len, dim = hidden_states.size()
        masks_sum = torch.sum(masks, dim=0)
        idx = masks_sum > 0
        idx = idx.unsqueeze(dim=0).expand_as(masks)
        masks = masks[idx]
        idx_ex = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states[idx_ex]
        hidden_states = hidden_states.view(bs, -1, dim)
        masks = masks.view(bs, -1)

        return hidden_states, masks, idx

    def generate(self, encoded_inputs, llm_input_ids=None, llm_attention_mask=None, **kwargs):
        bs = len(encoded_inputs)
        bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long,  device=llm_input_ids.device)
        bos_embedding = self.base_model_embedding_layer(bos)
        bos_embedding = bos_embedding.view(bs, 1, -1)
        if self.llm_bos_token_id is None:
            bos_mask = torch.zeros([bs, 1], dtype=torch.long,  device=llm_input_ids.device).cuda()
        else:
            bos_mask = torch.ones([bs, 1], dtype=torch.long,  device=llm_input_ids.device).cuda()
        
        input_embeds, attention_mask = self.embeddimg_model(encoded_inputs)
        input_embeds = self.mapper(input_embeds)

        input_embeds = torch.cat([bos_embedding, input_embeds], dim=1)
        attention_mask = torch.cat([bos_mask, attention_mask], dim=1)
        
        if llm_input_ids is not None:
            llm_input_embeds = self.base_model_embedding_layer(llm_input_ids)
            input_embeds = torch.cat([input_embeds, llm_input_embeds], dim=1)
            attention_mask = torch.cat([attention_mask, llm_attention_mask], dim=1)

        input_embeds, attention_mask, _ = self.squeeze_pad(input_embeds, attention_mask)

        generated_ids = self.base_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        return generated_ids

    def forward(self, encoded_inputs, labels=None, mask_label=None, input_ids_prompt=None, mask_prompt=None, **kwargs):
        bs = len(encoded_inputs)

        bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long).cuda()
        bos_embedding = self.base_model_embedding_layer(bos)
        bos_embedding = bos_embedding.view(bs, 1, -1)
        if self.llm_bos_token_id is None:
            mask = torch.zeros([bs, 1], dtype=torch.long).cuda()
        else:
            mask = torch.ones([bs, 1], dtype=torch.long).cuda()
        llm_input_embedding = bos_embedding
        llm_input_mask = mask

        encoder_last_hidden_state, attention_mask_mt = self.embeddimg_model(encoded_inputs)
        
        mt_encoder_hidden = []
        for i in self.encoder_aligner_config.encoder_layers:
            mt_encoder_hidden.append(encoder_last_hidden_state.hidden_states[i])

        adapter_states = self.encoder_aligner(mt_encoder_hidden)
        for i, index_layer in enumerate(self.base_model.peft_config["default"].adapter_layers):
            adapter_state = adapter_states[i]
            self.base_model.base_model.layers[index_layer].self_attn.update_adapter_states(adapter_state)
        
        mt_hidden_state = self.mapper(encoder_last_hidden_state)
        llm_input_embedding = torch.cat([llm_input_embedding, mt_hidden_state],
                                        dim=1)
        llm_input_mask = torch.cat([llm_input_mask, attention_mask_mt], dim=1)

        if input_ids_prompt is not None:
            hidden_states_prompt = self.base_model_embedding_layer(input_ids_prompt)
            llm_input_embedding = torch.cat([llm_input_embedding, hidden_states_prompt], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_prompt], dim=1)

        pad_labels = torch.full(
            (bs, llm_input_mask.shape[1]),
            -100,
            dtype=llm_input_mask.dtype,
            device=llm_input_mask.device,
        )
        
        label_embedding = self.base_model_embedding_layer(labels)
        llm_input_embedding = torch.cat([llm_input_embedding, label_embedding], dim=1)
        llm_input_mask = torch.cat([llm_input_mask, mask_label], dim=1)

        mask_label[:, -1] = 1.0
        labels[mask_label == 0] = -100
        
        labels = torch.cat([pad_labels, labels], dim=1)

        llm_input_embedding, llm_input_mask, cut_pad_idx \
            = self.squeeze_pad(llm_input_embedding, llm_input_mask)

        bs, seq_len = labels.size()
        labels = labels[cut_pad_idx]
        labels = labels.view(bs, -1)

        output = self.base_model(inputs_embeds=llm_input_embedding,
                                attention_mask=llm_input_mask,
                                labels=labels)
        return output


# Custom Model using PyTorch Lightning
class MPTrainer(pl.LightningModule):
    def __init__(self, trainer_args):
        super(MPTrainer, self).__init__()       

        self.tokenizer_llm = AutoTokenizer.from_pretrained(trainer_args['llm_model_name'], use_fast=True)
        self.tokenizer_llm.pad_token = self.tokenizer_llm.eos_token
        self.tokenizer_llm.padding_side = "left"
        
        self.model = MPTModel(
            embedding_model_base=trainer_args['embedding_model_base'],
            embedding_model_ext=trainer_args['embedding_model_ext'],
            num_embedding_tokens=trainer_args['context_token_count'],
            freeze_embedding=trainer_args['freeze_embedding'],
            llm_model_name=trainer_args['llm_model_name'],
            quantization_config=trainer_args['quantization_config'],
            lora_config=trainer_args['lora_config'],
            llm_bos_token_id=self.tokenizer_llm.bos_token_id,
            llm_pad_token_id=self.tokenizer_llm.pad_token_id
        )

        self.system_prompt = trainer_args['system_prompt']
        
        self.batch_size = trainer_args['batch_size']
        self.lr = trainer_args['lr']
        self.total_steps = train_args['total_steps']
        
        self.user_prompt_function = trainer_args['user_prompt_function']

        self.count_trainable_parameters()

    def forward(self, encoded_inputs, labels, label_mask, llm_input_ids=None, llm_attention_mask=None, **kwargs):
        output = self.model(
            encoded_inputs, labels, label_mask, llm_input_ids, llm_attention_mask, **kwargs
        )
        return output

    def training_step(self, batch, batch_idx):
        encoder_input, expected_output = batch
        
        if self.user_prompt_function is not None and self.system_prompt is not None:
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
        else:
            llm_inputs = {}
            llm_inputs['attention_mask'] = None
            llm_inputs['input_ids'] = None
            
        labels = [f"{sentence}{self.tokenizer_llm.eos_token}" for sentence in expected_output]
            
        llm_labels = self.tokenizer_llm(
            labels, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            add_special_tokens=False
        ).to(self.device)
        
        output = self(
            encoded_inputs = encoder_input,
            labels=llm_labels["input_ids"],
            label_mask = llm_labels['attention_mask'],
            llm_input_ids=llm_inputs['input_ids'],
            llm_attention_mask=llm_inputs['attention_mask'],
        )
        loss = output.loss
        
        self.log("train_loss", loss, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_step=True)
        return loss

    def count_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Percentage: {round((trainable_params/total_params)*100,2)}%")

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = DeepSpeedCPUAdam(params, lr=self.lr, betas=[0.8, 0.999], eps=1e-8, weight_decay=3e-7)
        
        warmup_ratio = 0.1  # 10% warm-up
        warmup_steps = int(self.total_steps * warmup_ratio)

        lr_scheduler = get_scheduler(
            name="cosine",  # could be 'linear', 'cosine', 'constant', etc.
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch' based on when you want to step the scheduler
            'frequency': 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def save_weights(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")

    def load_weights(self, file_path: str, map_location=None):
        if map_location is None:
            map_location = self.device
        self.model.load_state_dict(torch.load(file_path, map_location=map_location))
        print(f"Model weights loaded from {file_path}")

    def evaluate_dataset(self, encoder_inputs, batch_size=1, max_new_tokens=10):
        response = []
        with torch.no_grad():
            for i in tqdm(range(0, len(encoder_inputs), batch_size)):
                batch_q = encoder_inputs[i:i + batch_size]
                
                if self.user_prompt_function is not None and self.system_prompt is not None:
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
                else:
                    llm_inputs = {}
                    llm_inputs['attention_mask'] = None
                    llm_inputs['input_ids'] = None

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        encoded_inputs = batch_q,
                        llm_input_ids=llm_inputs['input_ids'],
                        llm_attention_mask=llm_inputs['attention_mask'],
                        do_sample=False,
                        max_new_tokens = max_new_tokens,
                        pad_token_id=self.tokenizer_llm.eos_token_id
                    )
                
                # Decode responses
                batch_responses = self.tokenizer_llm.batch_decode(generated_ids, skip_special_tokens=True)
                response.extend(batch_responses)
            
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


ds = load_dataset("facebook/xnli", "en")

train_ds = ds['train']

def create_encoder_prompt(example):
    premise = example['premise']
    hypothesis = example['hypothesis']
    labels = {0:'Entailment', 1:'Neutral', 2:'Contradiction'}
    sentence = f"Premise: {premise.strip()} Hypothesis: {hypothesis.strip()} Label:"
    label = f"{labels[int(example['label'])]}"

    return {
        'sentence': sentence,
        'conclusion': label
    }

train_ds = train_ds.map(create_encoder_prompt)

train_ds = train_ds[:100000]

# Access the train and test splits
# train_ds = ds.filter(lambda example: example['lang'] in ['en', 'sw'])

train_dataset = ExperimentDataset(
    train_ds['sentence'],
    train_ds['conclusion']
)


batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

llm_names = {
    1: "HuggingFaceTB/SmolLM2-135M-Instruct",
    2: "Qwen/Qwen2.5-0.5B-Instruct",
    3: "Qwen/Qwen2.5-1.5B-Instruct",
    4: "meta-llama/Llama-3.2-1B-Instruct",
    5: "meta-llama/Llama-2-7b-chat-hf",
    6: "meta-llama/Llama-3.1-8B-Instruct",
    7: "LLaMAX/LLaMAX2-7B-XNLI"
}

encoder_names = {
    1: "FacebookAI/xlm-roberta-base",
    2: "FacebookAI/xlm-roberta-large",
    3: "facebook/nllb-200-distilled-600M",
    4: "facebook/nllb-200-1.3B",
    5: "google/mt5-small",
    6: "google/mt5-base",
    7: "google/mt5-large",
    8: "DKYoon/mt5-small-lm-adapt",
    9: "DKYoon/mt5-large-lm-adapt",
    10: "DKYoon/mt5-xl-lm-adapt",
    11: "facebook/nllb-200-distilled-1.3B",
    12: "google/mt5-xl",
}

# Initialize Model and Trainer
llm_model_name = llm_names[5]
embedding_model_base = encoder_names[10]
embedding_model_ext = encoder_names[4]
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4"
# )
quantization_config=None
context_token_count =  -1
freeze_embedding = True
lr = 2e-5
epochs = 3

lora_config = LoraConfig(
    r=32,
    lora_alpha=128,  # Maintains an effective scaling factor of 4
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
    bias="all",
    task_type="CAUSAL_LM",
)

lora_config = None

train_args = {
    'llm_model_name': llm_model_name,
    'embedding_model_base': embedding_model_base,
    'embedding_model_ext': embedding_model_ext,
    'quantization_config': quantization_config,
    'context_token_count': context_token_count,
    'freeze_embedding': freeze_embedding,
    'system_prompt': None,
    'batch_size': batch_size,
    'lr': lr,
    'epochs': epochs,
    'lora_config': lora_config,
    'user_prompt_function': None
}

steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * epochs
train_args['total_steps'] = total_steps

model = MPTrainer(train_args)

print(model)

for name, param in model.model.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}, requires_grad={param.requires_grad}, shape={param.shape}")


# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=root_path,  # Directory to save checkpoints
    filename='final_checkpoint',  # Fixed filename for the last checkpoint
    save_top_k=1,  # Save only the final checkpoint
    save_last=False,  # Ensure the last checkpoint is saved
    every_n_epochs=epochs  # Save only at the last epoch
)

trainer = Trainer(
    accelerator="gpu",
    max_epochs=epochs,
    accumulate_grad_batches=4,
    strategy="deepspeed_stage_2_offload",
    precision="bf16",
    callbacks=[checkpoint_callback],
)

print('training')
# Train the Model
trainer.fit(
    model,
    train_dataloaders=train_loader,
    # val_dataloaders=val_loader
    # ckpt_path="/root/multilingual-p-tuning/final_checkpoint.ckpt/checkpoint/mp_rank_00_model_states.pt"
)

# final_val_loss = trainer.callback_metrics.get("val_loss", None)
final_train_loss = trainer.callback_metrics.get("train_loss", None)
print(final_train_loss)

model.save_weights(os.path.join(root_path, 'model.pth'))

model.eval()

model.to('cuda')

languages = ['en', 'ar', 'es', 'fr', 'hi', 'sw', 'th', 'ur', 'vi', 'zh']

for lang in languages:
    print(lang)
    ds = load_dataset("facebook/xnli", lang)
    test_ds = ds['test']
    test_ds = test_ds.map(create_encoder_prompt)
    res = model.evaluate_dataset(test_ds['sentence'], batch_size=batch_size, max_new_tokens=10)
    accuracy = accuracy_score(test_ds['conclusion'], res)
    f1 = f1_score(test_ds['conclusion'], res, average='weighted') 
    
    print(f"Accuracy {lang}: {accuracy}")
    print(f"F1 Score {lang}: {f1}")
    print(' ')
    