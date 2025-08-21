"""
BERT (Bidirectional Encoder Representations from Transformers) implementation in PyTorch.

Based on "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
by Devlin et al. https://arxiv.org/abs/1810.04805
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

from .base import BaseModel


class BertEmbeddings(nn.Module):
    """BERT Embeddings: token + position + segment embeddings."""
    
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int = 512,
                 type_vocab_size: int = 2, hidden_dropout_prob: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Register position_ids as buffer to make it available in device transfer
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, past_key_values_length: int = 0) -> torch.Tensor:
        
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """BERT Self-Attention mechanism."""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float = 0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"Hidden size ({hidden_size}) must be divisible by number of attention heads ({num_attention_heads})")
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    """BERT Self-Attention output projection."""
    
    def __init__(self, hidden_size: int, hidden_dropout_prob: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """BERT Attention layer combining self-attention and output projection."""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float = 0.1,
                 hidden_dropout_prob: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob, layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    """BERT Intermediate (feed-forward) layer."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """BERT Output layer with residual connection."""
    
    def __init__(self, intermediate_size: int, hidden_size: int, hidden_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """BERT Transformer layer."""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int,
                 attention_probs_dropout_prob: float = 0.1, hidden_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob,
                                     hidden_dropout_prob, layer_norm_eps)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size, hidden_dropout_prob, layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = self_attention_outputs[0]
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        outputs = (layer_output,) + self_attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    """BERT Encoder with multiple transformer layers."""
    
    def __init__(self, hidden_size: int, num_hidden_layers: int, num_attention_heads: int,
                 intermediate_size: int, attention_probs_dropout_prob: float = 0.1,
                 hidden_dropout_prob: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.layer = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size,
                     attention_probs_dropout_prob, hidden_dropout_prob, layer_norm_eps)
            for _ in range(num_hidden_layers)
        ])
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False, output_hidden_states: bool = False) -> Dict[str, Any]:
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions
        }


class BertPooler(nn.Module):
    """BERT Pooler for classification tasks."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pool the model by taking the hidden state of the first token ([CLS])
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERT(BaseModel):
    """
    BERT: Bidirectional Encoder Representations from Transformers.
    
    This implementation includes the core BERT model for various downstream tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.vocab_size = config.get('vocab_size', 30522)
        self.hidden_size = config.get('hidden_size', 768)
        self.num_hidden_layers = config.get('num_hidden_layers', 12)
        self.num_attention_heads = config.get('num_attention_heads', 12)
        self.intermediate_size = config.get('intermediate_size', 3072)
        self.max_position_embeddings = config.get('max_position_embeddings', 512)
        self.type_vocab_size = config.get('type_vocab_size', 2)
        self.attention_probs_dropout_prob = config.get('attention_probs_dropout_prob', 0.1)
        self.hidden_dropout_prob = config.get('hidden_dropout_prob', 0.1)
        self.layer_norm_eps = config.get('layer_norm_eps', 1e-12)
        
        # Model components
        self.embeddings = BertEmbeddings(
            self.vocab_size, self.hidden_size, self.max_position_embeddings,
            self.type_vocab_size, self.hidden_dropout_prob, self.layer_norm_eps
        )
        
        self.encoder = BertEncoder(
            self.hidden_size, self.num_hidden_layers, self.num_attention_heads,
            self.intermediate_size, self.attention_probs_dropout_prob,
            self.hidden_dropout_prob, self.layer_norm_eps
        )
        
        self.pooler = BertPooler(self.hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Create extended attention mask for self-attention.
        Makes very negative number so that the attention weight is close to zero.
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None,
                output_attentions: bool = False, output_hidden_states: bool = False) -> Dict[str, Any]:
        """
        Forward pass of BERT model.
        
        Args:
            input_ids: Token indices of input sequence
            attention_mask: Mask to avoid attention on padding tokens
            token_type_ids: Segment token indices
            position_ids: Indices of positions of each token
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
        
        Returns:
            Dictionary containing model outputs
        """
        input_shape = input_ids.size()
        device = input_ids.device
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        
        # Get extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        # Embedding layer
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids
        )
        
        # Encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = encoder_outputs['last_hidden_state']
        pooled_output = self.pooler(sequence_output)
        
        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
            'hidden_states': encoder_outputs['hidden_states'],
            'attentions': encoder_outputs['attentions']
        }


class BertForMaskedLM(BaseModel):
    """BERT model for Masked Language Modeling."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bert = BERT(config)
        self.cls = nn.Linear(config.get('hidden_size', 768), config.get('vocab_size', 30522))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:
        """Forward pass for masked language modeling."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        sequence_output = outputs['last_hidden_state']
        prediction_scores = self.cls(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.bert.vocab_size), labels.view(-1))
        
        return {
            'loss': masked_lm_loss,
            'logits': prediction_scores,
            'hidden_states': outputs['hidden_states'],
            'attentions': outputs['attentions']
        }


class BertForSequenceClassification(BaseModel):
    """BERT model for sequence classification."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_labels = config.get('num_labels', 2)
        self.bert = BERT(config)
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob', 0.1))
        self.classifier = nn.Linear(config.get('hidden_size', 768), self.num_labels)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:
        """Forward pass for sequence classification."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs['hidden_states'],
            'attentions': outputs['attentions']
        }