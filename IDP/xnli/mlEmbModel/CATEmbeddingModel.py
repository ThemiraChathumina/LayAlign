from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
from Configs import XNLIConfigs

class CrossLayerAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, 1, d_model))

    def forward(self, hidden_states):
        """
        hidden_states: [B, L, T, D]  (excluding embedding layer)
        Returns: [B, T, D]
        """
        B, L, T, D = hidden_states.size()

        # [B, T, L, D]
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        Q = self.query.expand(B, T, 1, D)
        K = V = hidden_states

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(D, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # [B, T, 1, D]

        return output.squeeze(2)  # [B, T, D]

class CATMultilingualEmbeddingModel(nn.Module):
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

        if embedding_model_base not in self.ALLOWED_MODELS:
            raise ValueError(f"Model is not in allowed models: {self.ALLOWED_MODELS}")
        
        self.embedding_model_base = AutoModel.from_pretrained(embedding_model_base)
        if "nllb" in embedding_model_base or "mt5" in embedding_model_base:
            self.embedding_model_base = self.embedding_model_base.encoder 

        self.freeze_embedding = freeze_embedding
        if freeze_embedding:
            for param in self.embedding_model_base.parameters():
                param.requires_grad = False
            
        self.tokenizer_base = AutoTokenizer.from_pretrained(embedding_model_base)

        self.max_seq_len = max_seq_len
        
        self.embedding_dim_base = self.embedding_model_base.config.hidden_size
        self.embedding_dim = self.embedding_dim_base

        
        
       
        self.configs = XNLIConfigs()

        self.langbridge_baseline = self.configs.baseline
        
        if not self.langbridge_baseline:
            self.softmax_gated = CrossLayerAttention(self.embedding_dim_base)

        
    def get_input_embeddings(self, model, input_ids):
        if "M2M" in model.__class__.__name__:
            return model.embed_tokens(input_ids)
        return model.get_input_embeddings()(input_ids)

    def mt_input_features(self, input_texts_m2m):
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
        m2m_pad_id = self.tokenizer_base.pad_token_id
        for input_ids_m2m_item, attention_mask_m2m_item in zip(input_ids_m2m, attention_mask_m2m):
            while len(input_ids_m2m_item) < max_len:
                input_ids_m2m_item.append(m2m_pad_id)
                attention_mask_m2m_item.append(0)
        input_ids_m2m = torch.tensor(input_ids_m2m, dtype=torch.long).cuda()
        attention_mask_m2m = torch.tensor(attention_mask_m2m, dtype=torch.long).cuda()
        return input_ids_m2m, attention_mask_m2m
    
    def get_last_hidden_states(self, encoded_inputs):
        input_ids, attention_mask = self.mt_input_features(encoded_inputs)
        outputs = self.embedding_model_base(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, attention_mask
        
    def cross_attended(self, encoded_inputs):
        input_ids, attention_mask = self.mt_input_features(encoded_inputs)
        outputs = self.embedding_model_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states_stacked = torch.stack(outputs.hidden_states[1:], dim=1)  # [B, L, T, D]
        fused_tokens = self.cross_layer_attention(hidden_states_stacked)  # [B, T, D]
        return fused_tokens, attention_mask
    
    def forward(self, encoded_inputs):
        if not self.langbridge_baseline:
            base_embeddings, base_attention_mask = self.cross_attended(
                encoded_inputs
            )
        else:
            base_embeddings, base_attention_mask = self.get_last_hidden_states(
                encoded_inputs
            )
        return base_embeddings, base_attention_mask
        
    
    def log_softmax_gate(self):
        pass
            
    def print_softmax_gate(self):
        pass

    def close_csv(self):
        pass