from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F


class LayerwiseTokenAggregator(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(hidden_dim))  # Learnable query

    def forward(self, layer_outputs):
        """
        layer_outputs: list of [batch_size, seq_len, hidden_dim], len = num_layers
        Returns: [batch_size, seq_len, hidden_dim] with attention across layers using learnable query
        """
        # Stack into [batch_size, num_layers, seq_len, hidden_dim]
        layer_stack = torch.stack(layer_outputs, dim=1)

        # Transpose to [batch_size, seq_len, num_layers, hidden_dim]
        layer_stack = layer_stack.transpose(1, 2)

        batch_size, seq_len, num_layers, hidden_dim = layer_stack.shape

        # Reshape for processing: [batch_size * seq_len, num_layers, hidden_dim]
        token_layer_seq = layer_stack.reshape(-1, num_layers, hidden_dim)

        # Expand learnable query: [batch_size * seq_len, 1, hidden_dim]
        query = self.query.expand(token_layer_seq.size(0), 1, hidden_dim)

        # Q = learnable query, K = V = layer-wise token sequence
        scores = torch.matmul(query, token_layer_seq.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size * seq_len, 1, num_layers]
        attended = torch.matmul(attn_weights, token_layer_seq)  # [batch_size * seq_len, 1, hidden_dim]

        # Remove extra dimension: [batch_size * seq_len, hidden_dim]
        token_representations = attended.squeeze(1)

        # Reshape back: [batch_size, seq_len, hidden_dim]
        return token_representations.view(batch_size, seq_len, hidden_dim)

class ATMultilingualEmbeddingModel(nn.Module):
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

        self.num_layers = self.embedding_model_base.config.num_hidden_layers  # Exclude embedding layer
        self.aggregator = LayerwiseTokenAggregator(self.embedding_dim_base, self.num_layers)

        
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
    
    def attention_gated(self, encoded_inputs):

        input_ids, attention_mask = self.mt_input_features(encoded_inputs)

        output = self.embedding_model_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = output.hidden_states[1:]  # Exclude embedding layer
        attention_gated = self.aggregator(hidden_states)
        return attention_gated, attention_mask

    def forward(self, encoded_inputs):
        base_embeddings, base_attention_mask = self.attention_gated(
            encoded_inputs
        )
        return base_embeddings, base_attention_mask
    
    def log_softmax_gate(self):
        pass
            
    def print_softmax_gate(self):
        pass

    def close_csv(self):
        pass