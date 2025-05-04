from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
from mlEmbModel import JSMMultilingualEmbeddingModel

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

class FusionBlock(nn.Module):
    def __init__(self, d_model, d_encoder, d_text, d_out, num_heads=8, num_layers=1, num_queries = 16):
        super(FusionBlock, self).__init__()
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.encoder_mapper = MLP(d_encoder, d_model)
        self.text_mapper = MLP(d_text, d_model)
        self.out_proj = MLP(d_model, d_out)
        qformer_layer = nn.TransformerDecoderLayer(d_model, num_heads)
        self.qformer = nn.TransformerDecoder(qformer_layer, num_layers)
    
    def forward(self, enc_embedding, enc_attention_mask, text_embedding, text_attention_mask):
        # enc_embedding: [batch_size, seq_len, d_encoder]
        # enc_attention_mask: [batch_size, seq_len]
    
        batch_size = enc_embedding.size(0)
        seq_len = enc_embedding.size(1)
    
        # Step 1: Map encoder embeddings to d_model dimension
        memory = self.encoder_mapper(enc_embedding)  # [batch_size, seq_len, d_model]
        text = self.text_mapper(text_embedding)
        
        # Step 2: Prepare learnable query tokens as target input
        # Expand queries for batch dimension
        tgt = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_queries, d_model]

        # Step 3: Create attention masks
        tgt_attention_mask = torch.ones((batch_size, tgt.size(1)), dtype=torch.bool, device=enc_embedding.device)  # [batch_size, num_queries]
        tgt_attention_mask = torch.cat([tgt_attention_mask, text_attention_mask], dim=1)
        tgt = torch.cat([tgt, text], dim=1)
        
        # Convert attention masks to key padding masks (True = masked)
        memory_key_padding_mask = ~enc_attention_mask.bool()  # [batch_size, seq_len]
        tgt_key_padding_mask = ~tgt_attention_mask.bool()     # [batch_size, num_queries]

        # Transformer expects shape: [tgt_len, batch_size, d_model] and [mem_len, batch_size, d_model]
        tgt = tgt.transpose(0, 1)        # [num_queries, batch_size, d_model]
        memory = memory.transpose(0, 1)  # [seq_len, batch_size, d_model]
    
        # Step 4: Run Q-Former decoder
        qformer_output = self.qformer(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # [num_queries, batch_size, d_model]
    
        # Step 5: Project Q-Former output to output dimension
        qformer_output = qformer_output.transpose(0, 1)  # [batch_size, num_queries, d_model]
        output = self.out_proj(qformer_output)           # [batch_size, num_queries, d_out]
    
        return output[:,:self.num_queries], tgt_attention_mask[:,:self.num_queries]
    
class MPTModel(nn.Module):
    def __init__(self, config):
        super(MPTModel, self).__init__()
        self.config = config  # Ensure there is a config attribute
        self.max_gen_len = config['max_gen_len']
        self.encoder_mt = JSMMultilingualEmbeddingModel(config['mt_path'], config['ext_path'], config['max_seq_len'])
        
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
        
    def print_gates(self):
        self.encoder_mt.print_softmax_gate()

    def clean_up(self):
        self.encoder_mt.close_csv()