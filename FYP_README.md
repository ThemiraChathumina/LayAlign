# Training and Evaluation Instructions

## To Train the Model

Run the following commands:

```bash
bash scripts/setup.sh
source multi-lingual/bin/activate
cd xnli_test
bash train.sh
```

## To Evaluate

Run the following command:

```bash
python eval.py
```

## Notes on Training

The current code trains for an experiment on softmax. To remove it and train using LangBridge, make the following changes in `model.py` under the `MultilingualEmbeddingModel` class:

1. Remove the following lines on softmax model initialization:

```python
# for softmax gating
num_layers = self.embedding_model_base.config.num_hidden_layers
self.layer_weights = LayerWeights(self.embedding_dim, num_layers)
```

2. Remove the following block:

```python
base_embeddings, base_attention_mask = self.softmax_gated(
    encoded_inputs, 
    self.tokenizer_base,
)
```

3. Uncomment the following block:

```python
base_embeddings, base_attention_mask = self.get_last_hidden_states(
    encoded_inputs, 
    self.embedding_model_base,
    self.tokenizer_base,
)
```
