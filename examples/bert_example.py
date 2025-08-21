"""
Example usage of BERT model.
"""

import torch
from models import BERT, BertForMaskedLM, BertForSequenceClassification
from utils import Config

def bert_example():
    """Example usage of BERT model."""
    print("=== BERT Example ===")
    
    # Base BERT configuration
    config = Config({
        'model_name': 'bert',
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'attention_probs_dropout_prob': 0.1,
        'hidden_dropout_prob': 0.1,
        'layer_norm_eps': 1e-12
    })
    
    # 1. Base BERT model
    print("\n1. Base BERT Model:")
    bert_model = BERT(config.to_dict())
    bert_model.eval()
    print(f"Model: {bert_model.model_name}")
    bert_model.summary()
    
    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, config.get('vocab_size'), (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    
    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  token_type_ids: {token_type_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    
    print(f"\nBase BERT outputs:")
    print(f"  last_hidden_state: {outputs['last_hidden_state'].shape}")
    print(f"  pooler_output: {outputs['pooler_output'].shape}")
    
    # 2. BERT for Masked Language Modeling
    print("\n2. BERT for Masked Language Modeling:")
    mlm_model = BertForMaskedLM(config.to_dict())
    mlm_model.eval()
    
    # Create masked input (replace some tokens with [MASK] token id = 103)
    masked_input_ids = input_ids.clone()
    mask_positions = torch.randint(0, seq_length, (batch_size, 5))  # Mask 5 tokens per sample
    for i in range(batch_size):
        masked_input_ids[i, mask_positions[i]] = 103  # [MASK] token
    
    labels = input_ids.clone()
    labels[labels != 103] = -100  # Only compute loss on masked tokens
    
    with torch.no_grad():
        mlm_outputs = mlm_model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
    
    print(f"MLM outputs:")
    print(f"  logits: {mlm_outputs['logits'].shape}")
    if mlm_outputs['loss'] is not None:
        print(f"  loss: {mlm_outputs['loss'].item():.6f}")
    
    # 3. BERT for Sequence Classification
    print("\n3. BERT for Sequence Classification:")
    cls_config = config.to_dict()
    cls_config['num_labels'] = 2  # Binary classification
    cls_model = BertForSequenceClassification(cls_config)
    cls_model.eval()
    
    # Create classification labels
    cls_labels = torch.randint(0, 2, (batch_size,))
    
    with torch.no_grad():
        cls_outputs = cls_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=cls_labels
        )
    
    print(f"Classification outputs:")
    print(f"  logits: {cls_outputs['logits'].shape}")
    if cls_outputs['loss'] is not None:
        print(f"  loss: {cls_outputs['loss'].item():.6f}")
    
    print("\n=== BERT Example Completed ===\n")

if __name__ == '__main__':
    bert_example()