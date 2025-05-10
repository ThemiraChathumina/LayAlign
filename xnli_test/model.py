from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
import random

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

    
class LayerWeights(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(LayerWeights, self).__init__()
        # <-- enable batch_first so nested_tensor path is used and warning disappears
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=1,
            batch_first=True
        )
        self.transformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )
        self.layer_pos_emb = nn.Parameter(torch.randn(num_layers, hidden_size))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        self.layer_weights = nn.Linear(hidden_size, num_layers)

    def forward(self, x):
        # x: [B, num_layers, hidden]
        x = x + self.layer_pos_emb                  # [B, num_layers, hidden]
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = self.transformerEncoder(x)              # [B, num_layers+1, hidden]
        x = x[:, 0, :]                              # [B, hidden]
        x = self.layer_weights(x)                   # [B, num_layers]
        return F.softmax(x, dim=-1)

class MultilingualEmbeddingModel(nn.Module):
    ALLOWED_MODELS = {
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "facebook/nllb-200-distilled-600M",
        "facebook/nllb-200-1.3B",
        "google/mt5-small",
        "google/mt5-base",
        "google/mt5-large",
        "google/mt5-xl",
        "DKYoon/mt5-small-lm-adapt",
        "DKYoon/mt5-large-lm-adapt",
        "DKYoon/mt5-xl-lm-adapt",
        "facebook/nllb-200-distilled-1.3B"
    }
    
    def __init__(self, embedding_model_base, embedding_model_ext, max_seq_len, freeze_embedding = True):
        super().__init__()

        if embedding_model_base not in self.ALLOWED_MODELS or embedding_model_ext not in self.ALLOWED_MODELS:
            raise ValueError(f"Model is not in allowed models: {self.ALLOWED_MODELS}")
        
        self.embedding_model_base = AutoModel.from_pretrained(embedding_model_base)
        if "nllb" in embedding_model_base or "mt5" in embedding_model_base:
            self.embedding_model_base = self.embedding_model_base.encoder 

        # self.embedding_model_ext = AutoModel.from_pretrained(embedding_model_ext)
        # if "nllb" in embedding_model_ext or "mt5" in embedding_model_ext:
        #     self.embedding_model_ext = self.embedding_model_ext.encoder 

        self.freeze_embedding = freeze_embedding
        if freeze_embedding:
            for param in self.embedding_model_base.parameters():
                param.requires_grad = False
            # for param in self.embedding_model_ext.parameters():
            #     param.requires_grad = False
            
        self.tokenizer_base = AutoTokenizer.from_pretrained(embedding_model_base)
        # self.tokenizer_ext = AutoTokenizer.from_pretrained(embedding_model_ext)
        
        self.embedding_dim_base = self.embedding_model_base.config.hidden_size
        # self.embedding_dim_ext = self.embedding_model_ext.config.hidden_size
        self.embedding_dim = self.embedding_dim_base

        self.max_seq_len = max_seq_len
        
        # for softmax gating
        num_layers = self.embedding_model_base.config.num_hidden_layers
        self.layer_weights = LayerWeights(self.embedding_dim, num_layers)
        

    def get_input_embeddings(self, model, input_ids):
        if "M2M" in model.__class__.__name__:
            return model.embed_tokens(input_ids)
        return model.get_input_embeddings()(input_ids)
    
    def get_last_hidden_states(self, encoded_inputs, model, tokenizer):
        input_ids, attention_mask = self.mt_input_features(encoded_inputs, tokenizer)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, attention_mask

    def mt_input_features(self, input_texts_m2m, tokenizer):
        input_ids_m2m, attention_mask_m2m = [], []
        for input_text_m2m in input_texts_m2m:
            encoding_m2m = self.tokenizer_base(input_text_m2m,
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
    
    def softmax_gated(self, encoded_inputs, tokenizer):
        # 1) tokenize & get attention mask
        input_ids, attention_mask = self.mt_input_features(encoded_inputs, tokenizer)

        # 2) embed + encoder pass with all hidden‐states
        inputs_embeds = self.get_input_embeddings(self.embedding_model_base, input_ids)
        outputs = self.embedding_model_base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # 3) stack only the transformer layers (skip the embedding layer)
        #    hidden_states: tuple length = num_layers+1
        #    layer_hs: [batch, num_layers, seq_len, hidden_dim]
        layer_hs = torch.stack(outputs.hidden_states[1:], dim=1)
        B, L, T, D = layer_hs.size()

        # 4) rearrange to [B*T, L, D] so each token is an example
        layered = layer_hs.permute(0, 2, 1, 3)           # [B, T, L, D]
        x_flat = layered.reshape(B * T, L, D)           # [B*T, L, D]

        # 5) select only real tokens
        mask_flat = attention_mask.reshape(-1).bool()   # [B*T]
        valid_x = x_flat[mask_flat]                     # [N_valid, L, D]

        # 6) compute per‐token layer‐weights only for valid tokens
        valid_w = self.layer_weights(valid_x)           # [N_valid, L]

        # 7) scatter back into full-weight tensor (zeros for padding tokens)
        weights_flat = x_flat.new_zeros(B * T, L)       # [B*T, L]
        weights_flat[mask_flat] = valid_w               # fill only real tokens
        weights = weights_flat.view(B, T, L)            # [B, T, L]

        # 8) weighted sum over layers → [B, T, D]
        gated_flat = (x_flat * weights_flat.unsqueeze(-1)).sum(dim=1)  # [B*T, D]
        gated = gated_flat.view(B, T, D)                              # [B, T, D]

        # 9) ensure padding stays zero
        gated = gated * attention_mask.unsqueeze(-1)                  # [B, T, D]

        return gated, attention_mask

    
    def forward(self, encoded_inputs):
        base_embeddings, base_attention_mask = self.softmax_gated(
            encoded_inputs, 
            self.tokenizer_base,
        )
        
        # for baseline langbridge
        # base_embeddings, base_attention_mask = self.get_last_hidden_states(
        #     encoded_inputs, 
        #     self.embedding_model_base,
        #     self.tokenizer_base,
        # )

        return base_embeddings, base_attention_mask
    

class MPTModel(nn.Module):
    def __init__(self, config):
        super(MPTModel, self).__init__()
        self.config = config  # Ensure there is a config attribute
        self.max_gen_len = config['max_gen_len']
        self.encoder_mt = MultilingualEmbeddingModel(config['mt_path'], config['ext_path'], config['max_seq_len'])
        
        model_llm = AutoModelForCausalLM.from_pretrained(config['llm_path'])

        self.model_llm = model_llm

        self.llm_embedding_layer = self.model_llm.get_input_embeddings()
        for name, parameter in self.model_llm.named_parameters():
            parameter.requires_grad = False

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

    def forward(self, encoded_inputs,
                labels=None, mask_label=None, input_ids_prompt=None, mask_prompt=None):
        end_boundary = self.mapping.get_embed()
        bs = len(encoded_inputs)
        end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)])

        bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long).cuda()
        bos_embedding = self.llm_embedding_layer(bos)
        bos_embedding = bos_embedding.view(bs, 1, -1)
        mask = torch.ones([bs, 1], dtype=torch.long).cuda()
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