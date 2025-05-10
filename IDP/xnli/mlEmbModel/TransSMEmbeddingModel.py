from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F

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

class TransSMMultilingualEmbeddingModel(nn.Module):
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
    
    def log_softmax_gate(self):
        pass
            
    def print_softmax_gate(self):
        pass

    def close_csv(self):
        pass