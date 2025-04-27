from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
from transformers.models.mt5.modeling_mt5 import MT5Attention
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig


class MLP(nn.Module):
    def __init__(self, mt_dim, llm_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(mt_dim, mt_dim * 2)
        self.linear2 = nn.Linear(mt_dim * 2, llm_dim)
        self.relu = nn.ReLU()
    def forward(self, mt_hidden_state):
        output = self.linear1(mt_hidden_state)
        output = self.relu(output)
        output = self.linear2(output)
        return output

class Mapping(nn.Module):
    def __init__(self, mt_dim, llm_dim):
        super(Mapping, self).__init__()
        self.mlp = MLP(mt_dim, llm_dim)
        self.end_boundary = nn.Parameter(
            torch.zeros(1, 1, llm_dim), requires_grad=True
        )
    def forward(self, hidden_states):
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def get_embed(self):
        return self.end_boundary

class GatedMT5Attention(MT5Attention):
    def __init__(self, config, prompt_length, hidden_size, pretrained_attention=None, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        if pretrained_attention is not None:
            self.load_state_dict(pretrained_attention.state_dict(), strict=False)
        self.gate = nn.Parameter(torch.zeros(self.n_heads))
        self.prompts = nn.Parameter(torch.zeros(prompt_length, hidden_size))

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states.shape[:2]
        
        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        prompt_embeds = self.prompts
        prompt_length = prompt_embeds.size(0)
        prompt = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = torch.cat([prompt, hidden_states], dim=1)


        current_states = hidden_states
        key_states = self.k(current_states)
        value_states = self.v(current_states)
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        prompt_key = key_states[:, :, :prompt_length, :]
        token_key = key_states[:, :, prompt_length:, :]
        prompt_value = value_states[:, :, :prompt_length, :]
        token_value = value_states[:, :, prompt_length:, :]

        prompt_scores = torch.matmul(query_states, prompt_key.transpose(3, 2))
        scores = torch.matmul(query_states, token_key.transpose(3, 2))
        
        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        # scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked

        attn_weights_prompts = nn.functional.softmax(prompt_scores.float(), dim=-1).type_as(prompt_scores)
        gated_prompt_scores = torch.tanh(self.gate).unsqueeze(0).unsqueeze(2).unsqueeze(3) * attn_weights_prompts
        
        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_weights = torch.cat([gated_prompt_scores, attn_weights], dim=-1)
        value_states = torch.cat([prompt_value, token_value], dim=2)
        
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class MultilingualEmbeddingModel(nn.Module):
    ALLOWED_MODELS = {
        "google/mt5-small",
        "google/mt5-base",
        "google/mt5-large",
        "google/mt5-xl",
        "DKYoon/mt5-small-lm-adapt",
        "DKYoon/mt5-large-lm-adapt",
        "DKYoon/mt5-xl-lm-adapt",
    }
    
    def __init__(self, embedding_model, max_seq_len):
        super().__init__()

        if embedding_model not in self.ALLOWED_MODELS:
            raise ValueError(f"Model is not in allowed models: {self.ALLOWED_MODELS}")
        
        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.embedding_model = self.embedding_model.encoder 
            
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        
        self.embedding_dim = self.embedding_model.config.hidden_size

        self.max_seq_len = max_seq_len

        self.num_layers = len(self.embedding_model.block)

        self.prompt_length = 10

        prompt_layers = [-4,-3,-2,-1]
        
        for idx in prompt_layers:
            attention_weights = self.embedding_model.block[idx].layer[0].SelfAttention
            self.embedding_model.block[idx].layer[0].SelfAttention = GatedMT5Attention(self.embedding_model.config, self.prompt_length, self.embedding_dim, pretrained_attention=attention_weights)

        for name, param in self.named_parameters():
            if "prompts" in name or "gate" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def get_input_embeddings(self, model, input_ids):
        return model.get_input_embeddings()(input_ids)
    
    def get_last_hidden_states(self, encoded_inputs, model, tokenizer):
        input_ids, attention_mask = self.mt_input_features(encoded_inputs, tokenizer)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, attention_mask

    def mt_input_features(self, input_texts_m2m, tokenizer):
        input_ids_m2m, attention_mask_m2m = [], []
        for input_text_m2m in input_texts_m2m:
            encoding_m2m = tokenizer(input_text_m2m,
                                         padding='longest',
                                         max_length=self.max_seq_len,
                                         truncation=True)
            input_ids_m2m_item = encoding_m2m.input_ids
            attention_mask_m2m_item = encoding_m2m.attention_mask
            input_ids_m2m.append(input_ids_m2m_item)
            attention_mask_m2m.append(attention_mask_m2m_item)
        max_len = max([len(item) for item in input_ids_m2m])
        m2m_pad_id = tokenizer.pad_token_id
        for input_ids_m2m_item, attention_mask_m2m_item in zip(input_ids_m2m, attention_mask_m2m):
            while len(input_ids_m2m_item) < max_len:
                input_ids_m2m_item.append(m2m_pad_id)
                attention_mask_m2m_item.append(0)
        input_ids_m2m = torch.tensor(input_ids_m2m, dtype=torch.long).cuda()
        attention_mask_m2m = torch.tensor(attention_mask_m2m, dtype=torch.long).cuda()
        return input_ids_m2m, attention_mask_m2m
    
    def forward(self, encoded_inputs):
        embeddings, attention_mask = self.get_last_hidden_states(
            encoded_inputs, 
            self.embedding_model, 
            self.tokenizer,
        )
        
        return embeddings, attention_mask

class MPTModel(nn.Module):
    def __init__(self, config):
        super(MPTModel, self).__init__()
        self.config = config  # Ensure there is a config attribute
        self.max_gen_len = config['max_gen_len']
        self.encoder_mt = MultilingualEmbeddingModel(config['mt_path'], config['max_seq_len'])
        
        model_llm = AutoModelForCausalLM.from_pretrained(config['llm_path'])

        self.model_llm = model_llm

        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=128,  # Maintains an effective scaling factor of 4
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="all",
            task_type="CAUSAL_LM",
        )

        self.llm_embedding_layer = self.model_llm.get_input_embeddings()
        for name, parameter in self.model_llm.named_parameters():
            parameter.requires_grad = False

        # self.model_llm = get_peft_model(self.model_llm, self.lora_config)

        # for name, param in self.model_llm.named_parameters():
        #     if "lora" in name:
        #         param.requires_grad = True  

        d_model = self.encoder_mt.embedding_dim
        self.mapping = Mapping(d_model, model_llm.config.hidden_size)
        self.llm_pad_token_id = config['llm_pad_token_id']
        self.llm_bos_token_id = config['llm_bos_token_id']
        print('mapping layer size:', sum(param.numel() for param in self.mapping.parameters()) / 1000000)

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

    def merge_loara(self):
        self.model_llm = self.model_llm.merge_and_unload()
        self.model_llm.save_pretrained('/kaggle/working/llama_1b_xcsqa/')

    def forward(self, encoded_inputs,
                labels=None, mask_label=None, input_ids_prompt=None, mask_prompt=None):
        end_boundary = self.mapping.get_embed()
        bs = len(encoded_inputs)
        end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)])

        if self.llm_bos_token_id is None:
            bos = torch.tensor([self.llm_pad_token_id for i in range(bs)], dtype=torch.long).cuda()
            mask = torch.zeros([bs, 1], dtype=torch.long).cuda()
        else:
            bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long).cuda()
            mask = torch.ones([bs, 1], dtype=torch.long).cuda()
        bos_embedding = self.llm_embedding_layer(bos)
        bos_embedding = bos_embedding.view(bs, 1, -1)
        llm_input_embedding = bos_embedding
        llm_input_mask = mask

        mt_encoder_outputs, attention_mask_mt = self.encoder_mt(encoded_inputs)
        
        mt_hidden_state = self.mapping(mt_encoder_outputs)
        llm_input_embedding = torch.cat([llm_input_embedding, mt_hidden_state, end_boundary],
                                        dim=1)
        llm_input_mask = torch.cat([llm_input_mask, attention_mask_mt, mask], dim=1)

        if input_ids_prompt is not None:

            hidden_states_prompt = self.llm_embedding_layer(input_ids_prompt)
            llm_input_embedding = torch.cat([llm_input_embedding, hidden_states_prompt], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_prompt], dim=1)
        if labels is not None:
            pad_labels = llm_input_mask * -100 + (1 - llm_input_mask) * -100
            label_embedding = self.llm_embedding_layer(labels)
            llm_input_embedding = torch.cat([llm_input_embedding, label_embedding], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_label], dim=1)
            labels = labels * mask_label - 100 * (1 - mask_label)
            labels = torch.cat([pad_labels, labels], dim=1)

        llm_input_embedding, llm_input_mask, cut_pad_idx \
            = self.squeeze_pad(llm_input_embedding, llm_input_mask)

        if labels is None:
            generate_ids = self.model_llm.generate(inputs_embeds=llm_input_embedding,
                                                   attention_mask=llm_input_mask,
                                                   max_new_tokens=self.max_gen_len,
                                                   pad_token_id=self.llm_pad_token_id,
                                                   do_sample=False)
            return generate_ids
        else:
            bs, seq_len = labels.size()
            labels = labels[cut_pad_idx]
            labels = labels.view(bs, -1)
            output = self.model_llm(inputs_embeds=llm_input_embedding,
                                    attention_mask=llm_input_mask,
                                    labels=labels)
            return output.loss