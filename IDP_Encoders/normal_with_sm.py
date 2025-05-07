from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F

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
        
        self.num_embedding_tokens = 1
        
        self.num_layers = self.embedding_model_base.config.num_hidden_layers + 1
        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_embedding_tokens, self.embedding_dim)
        )  # [1, Q, D]

        # Scalar weights for each layer (same for all tokens)
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers))  # [L]

        # self.fusion_block = FusionBlock(
        #     d_model = self.embedding_dim_base,
        #     d_encoder = self.embedding_dim_ext,
        #     d_text = self.embedding_dim_base,
        #     d_out = self.embedding_dim
        # )
        # self.num_embedding_tokens = 2
        # self.learnable_queries_base = nn.Parameter(torch.randn(self.num_embedding_tokens, self.embedding_dim_base))
        
    def get_input_embeddings(self, model, input_ids):
        if "M2M" in model.__class__.__name__:
            return model.embed_tokens(input_ids)
        return model.get_input_embeddings()(input_ids)
    
    def get_last_hidden_states(self, encoded_inputs, model, tokenizer):
        # input_ids, attention_mask = self.mt_input_features(encoded_inputs, tokenizer)
        # batch_size = input_ids.shape[0]
        
        # inputs_embeds = self.get_input_embeddings(model, input_ids)  # [B, L, D]
        
        # queries = self.learnable_queries_base.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Q, D]
        
        # combined_inputs = torch.cat([queries, inputs_embeds], dim=1)  # [B, Q+L, D]
        
        # query_mask = torch.ones(batch_size, self.num_embedding_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
        # combined_attention_mask = torch.cat([query_mask, attention_mask], dim=1)  # [B, Q+L]
        
        # outputs = model(inputs_embeds=combined_inputs, attention_mask=combined_attention_mask)
        
        # return outputs.last_hidden_state, combined_attention_mask
        
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
    
    def softmax_gated(self, encoded_inputs, tokenizer, queries=None):
        input_ids, attention_mask = self.mt_input_features(encoded_inputs, tokenizer)

        inputs_embeds = self.get_input_embeddings(self.embedding_model_base, input_ids)  # [B, T, D]
        batch_size = inputs_embeds.size(0)

        
        # Expand and prepend learnable query tokens
        if queries is None:
            raise ValueError("Query tokens must be provided.")
        num_query_tokens = queries.size(1)
        expanded_queries = queries.expand(batch_size, -1, -1)  # [B, Q, D]

        # Prepend query tokens to embeddings
        inputs_embeds = torch.cat([expanded_queries, inputs_embeds], dim=1)  # [B, Q+T, D]

        # Adjust attention mask accordingly
        query_mask = torch.ones(batch_size, num_query_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([query_mask, attention_mask], dim=1)  # [B, Q+T]

        # Forward through encoder with all hidden states
        outputs = self.embedding_model_base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get all hidden states: tuple of length L+1 each [B, Q+T, D]
        hidden_states = outputs.hidden_states  # len = num_layers + 1

        # Extract query token representations from each layer
        # => [B, L, Q, D]
        per_layer_query_states = torch.stack([
            layer[:, :num_query_tokens, :] for layer in hidden_states
        ], dim=1)

        # Learnable layer weights
        norm_weights = F.softmax(self.layer_weights, dim=0)  # [L]
        norm_weights = norm_weights.view(1, -1, 1, 1)         # [1, L, 1, 1]

        # Weighted sum across layers
        fused_queries = torch.sum(per_layer_query_states * norm_weights, dim=1)  # [B, Q, D]
        # Extract last hidden state and remove query token positions
        last_hidden = outputs.last_hidden_state[:, num_query_tokens:, :]  # [B, T, D]

        # Concatenate fused queries + last layer's token representations
        full_embeddings = torch.cat([fused_queries, last_hidden], dim=1)  # [B, Q+T, D]

        return full_embeddings, attention_mask
    
    def forward(self, encoded_inputs):
        base_embeddings, base_attention_mask = self.softmax_gated(
            encoded_inputs, 
            self.tokenizer_base,
            self.query_tokens
        )

        # ext_embeddings, ext_attention_mask = self.get_last_hidden_states(
        #     encoded_inputs, 
        #     self.embedding_model_ext, 
        #     self.tokenizer_ext,
        # )

        # fused_embeddings, fused_attention_mask = self.fusion_block(ext_embeddings, ext_attention_mask, base_embeddings, base_attention_mask)

        # fused_embeddings = torch.cat([fused_embeddings, base_embeddings], dim=1)
        # fused_attention_mask = torch.cat([fused_attention_mask, base_attention_mask], dim=1)

        return base_embeddings, base_attention_mask