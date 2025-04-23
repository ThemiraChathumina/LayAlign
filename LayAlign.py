
from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from torch import nn
from transformers import LlamaConfig
from layer_wise_aligner import EncoderAligner
from peft import get_peft_model, AdaptionPromptConfig

class LayAlignConfig(LlamaConfig):
    def __init__(self, mt_path, llm_path, max_gen_len, llm_bos_token_id, llm_pad_token_id, encoder_aligner_config, augmentation, **kwargs):
        super().__init__(**kwargs)
        self.mt_path = mt_path
        self.llm_path = llm_path
        self.max_gen_len = max_gen_len
        self.llm_bos_token_id = llm_bos_token_id
        self.llm_pad_token_id = llm_pad_token_id
        self.encoder_aligner_config = encoder_aligner_config
        self.augmentation = augmentation


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
    
    def __init__(self, embedding_model_base, freeze_embedding = True):
        super().__init__()

        if embedding_model_base not in self.ALLOWED_MODELS:
            raise ValueError(f"Model is not in allowed models: {self.ALLOWED_MODELS}")
        
        self.embedding_model_base = AutoModel.from_pretrained(embedding_model_base)
        if "nllb" in embedding_model_base or "mt5" in embedding_model_base:
            self.embedding_model_base = self.embedding_model_base.encoder 

        self.embedding_dim_base = self.embedding_model_base.config.hidden_size
        self.embedding_dim = self.embedding_dim_base

        self.freeze_embedding = freeze_embedding
        if freeze_embedding:
            for param in self.embedding_model_base.parameters():
                param.requires_grad = False
                
    def get_input_embeddings(self, model, input_ids):
        if "M2M" in model.__class__.__name__:
            return model.embed_tokens(input_ids)
        return model.get_input_embeddings()(input_ids)
    
    def get_last_hidden_states(self, input_ids, attention_mask , model):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs, attention_mask
    
    def forward(self, input_ids, attention_mask):
        base_embeddings, base_attention_mask = self.get_last_hidden_states(
            input_ids, 
            attention_mask,
            self.embedding_model_base, 
        )
        
        return base_embeddings, base_attention_mask

class LayAlign(nn.Module):
    def __init__(self, config: LayAlignConfig, freeze=True, encoder_freeze=True):
        super(LayAlign, self).__init__()
        self.config = config  # Ensure there is a config attribute
        self.max_gen_len = config.max_gen_len
        self.encoder_mt = MultilingualEmbeddingModel(config.mt_path, freeze_embedding=encoder_freeze)
        
        model_llm = AutoModelForCausalLM.from_pretrained(config.llm_path)

        # peft_config = AdaptionPromptConfig(
        #     adapter_layers = self.config.encoder_aligner_config.language_layers
        # )
        
        # model_llm = get_peft_model(model_llm, peft_config).base_model
        self.model_llm = model_llm
        # if config.augmentation and not freeze:
        #     for name, parameter in self.model_llm.named_parameters():
        #         parameter.requires_grad = True
        
        self.llm_embedding_layer = self.model_llm.get_input_embeddings()
        for name, parameter in self.model_llm.named_parameters():
            parameter.requires_grad = False
        # if 'bert' in config.mt_path or 'Qwen' in config.mt_path:
        #     d_model = model_mt.config.hidden_size
        # elif 'GPT' in config.mt_path:
        #     d_model = model_mt.config.n_embd
        # else:
        d_model = self.encoder_mt.embedding_dim
        self.mapping = Mapping(d_model, model_llm.config.hidden_size)
        # self.encoder_aligner = EncoderAligner(config.encoder_aligner_config)
        self.llm_pad_token_id = config.llm_pad_token_id
        self.llm_bos_token_id = config.llm_bos_token_id
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

    def forward(self, input_ids_mt, attention_mask_mt,
                labels=None, mask_label=None, input_ids_prompt=None, mask_prompt=None):
        end_boundary = self.mapping.get_embed()
        bs = input_ids_mt.size(0)
        end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)])

        bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long).cuda()
        bos_embedding = self.llm_embedding_layer(bos)
        bos_embedding = bos_embedding.view(bs, 1, -1)
        mask = torch.ones([bs, 1], dtype=torch.long).cuda()
        llm_input_embedding = bos_embedding
        llm_input_mask = mask

        mt_encoder_outputs, attention_mask_mt = self.encoder_mt(input_ids=input_ids_mt,
                                             attention_mask=attention_mask_mt)
        
        # mt_encoder_hidden = []
        # for i in self.config.encoder_aligner_config.encoder_layers:
        #     mt_encoder_hidden.append(mt_encoder_outputs.hidden_states[i])

        # adapter_states = self.encoder_aligner(mt_encoder_hidden)
        # for i, index_layer in enumerate(self.model_llm.peft_config["default"].adapter_layers):
        #     adapter_state = adapter_states[i]
        #     self.model_llm.base_model.layers[index_layer].self_attn.update_adapter_states(adapter_state)

        encoder_last_hidden_state = mt_encoder_outputs[0]
        mt_hidden_state = self.mapping(encoder_last_hidden_state)
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

if __name__ == "__main__":
    from transformers import AutoTokenizer
    device = "cuda:2"
    text = "hello world"
    tokenizer_mt = AutoTokenizer.from_pretrained("google/mt5-xl")
    model_mt = AutoModel.from_pretrained("google/mt5-xl", torch_dtype=torch.float16, output_hidden_states=True).to(device)
    input = tokenizer_mt(text, return_tensors="pt").to(device)
    input = {k: v.to(device) for k, v in input.items()}
    with torch.no_grad():
        output = model_mt.encoder(**input)
    #hidden_states: tuple, len=24, hidden_states[0].shape: 1,3,2048(batch, token, )
    hidden_states = output.hidden_states[1:]
    print(output)