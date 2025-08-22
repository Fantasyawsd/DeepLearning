"""
BERT模型实现

BERT (Bidirectional Encoder Representations from Transformers) 是谷歌在2018年提出的
双向编码器表示模型，通过掩码语言模型预训练获得强大的语言理解能力。

论文: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)
作者: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import Dict, Any, Optional, Tuple

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from shared.base_model import BaseModel


class BertEmbeddings(nn.Module):
    """BERT嵌入层：词嵌入 + 位置嵌入 + token类型嵌入"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.vocab_size = config.get('vocab_size', 30522)
        self.hidden_size = config.get('hidden_size', 768)
        self.max_position_embeddings = config.get('max_position_embeddings', 512)
        self.type_vocab_size = config.get('type_vocab_size', 2)
        self.dropout = config.get('hidden_dropout_prob', 0.1)
        
        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(self.type_vocab_size, self.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(self.dropout)
        
        # position_ids 为注册的缓冲区
        self.register_buffer("position_ids", torch.arange(self.max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
        """
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertSelfAttention(nn.Module):
    """BERT自注意力层"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.num_attention_heads = config.get('num_attention_heads', 12)
        self.attention_head_size = config.get('hidden_size', 768) // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.get('hidden_size', 768), self.all_head_size)
        self.key = nn.Linear(config.get('hidden_size', 768), self.all_head_size)
        self.value = nn.Linear(config.get('hidden_size', 768), self.all_head_size)
        
        self.dropout = nn.Dropout(config.get('attention_probs_dropout_prob', 0.1))
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 计算注意力分数
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
        
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    """BERT自注意力输出层"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dense = nn.Linear(config.get('hidden_size', 768), config.get('hidden_size', 768))
        self.LayerNorm = nn.LayerNorm(config.get('hidden_size', 768), eps=config.get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob', 0.1))
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """BERT注意力模块"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output, self_outputs[1]  # 返回输出和注意力权重


class BertIntermediate(nn.Module):
    """BERT中间层（FFN的第一部分）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dense = nn.Linear(config.get('hidden_size', 768), config.get('intermediate_size', 3072))
        self.intermediate_act_fn = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """BERT输出层（FFN的第二部分）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dense = nn.Linear(config.get('intermediate_size', 3072), config.get('hidden_size', 768))
        self.LayerNorm = nn.LayerNorm(config.get('hidden_size', 768), eps=config.get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob', 0.1))
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """BERT Transformer层"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output, attention_outputs[1]  # 返回输出和注意力权重


class BertEncoder(nn.Module):
    """BERT编码器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.get('num_hidden_layers', 12))])
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        all_attentions = () if output_attentions else None
        
        for layer_module in self.layer:
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        return hidden_states, all_attentions


class BertPooler(nn.Module):
    """BERT池化层，用于序列级别的表示"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dense = nn.Linear(config.get('hidden_size', 768), config.get('hidden_size', 768))
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取[CLS] token的表示
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(BaseModel):
    """
    BERT基础模型
    
    Args:
        config: 模型配置字典
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.get('initializer_range', 0.02))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.get('initializer_range', 0.02))
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        self.apply(_init_weights)
    
    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """扩展注意力掩码"""
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"注意力掩码维度错误: {attention_mask.dim()}")
        
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    @property
    def dtype(self) -> torch.dtype:
        """获取模型数据类型"""
        return next(self.parameters()).dtype
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            output_attentions: 是否输出注意力权重
        
        Returns:
            last_hidden_state: [batch_size, seq_len, hidden_size]
            pooler_output: [batch_size, hidden_size]
            attentions: 注意力权重 (可选)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, output_attentions)
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        
        return sequence_output, pooled_output, encoder_outputs[1]


class BertForMaskedLM(BertModel):
    """用于掩码语言模型的BERT"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.cls = BertLMPredictionHead(config)
        self.init_weights()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            labels: [batch_size, seq_len] 掩码位置的真实标签，-100表示不计算损失
        """
        outputs = super().forward(input_ids, attention_mask, token_type_ids)
        sequence_output = outputs[0]
        
        prediction_scores = self.cls(sequence_output)
        
        result = {
            'logits': prediction_scores,
            'hidden_states': sequence_output,
            'pooler_output': outputs[1]
        }
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.get('vocab_size', 30522)), labels.view(-1))
            result['loss'] = masked_lm_loss
        
        return result


class BertForSequenceClassification(BertModel):
    """用于序列分类的BERT"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_labels = config.get('num_labels', 2)
        self.classifier = nn.Linear(config.get('hidden_size', 768), self.num_labels)
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob', 0.1))
        
        self.init_weights()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            labels: [batch_size] 分类标签
        """
        outputs = super().forward(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {
            'logits': logits,
            'hidden_states': outputs[0],
            'pooler_output': pooled_output
        }
        
        if labels is not None:
            if self.num_labels == 1:
                # 回归任务
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # 分类任务
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            result['loss'] = loss
        
        return result


class BertLMPredictionHead(nn.Module):
    """BERT语言模型预测头"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.get('hidden_size', 768), config.get('vocab_size', 30522), bias=False)
        self.bias = nn.Parameter(torch.zeros(config.get('vocab_size', 30522)))
        self.decoder.bias = self.bias
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    """BERT预测头变换"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dense = nn.Linear(config.get('hidden_size', 768), config.get('hidden_size', 768))
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.get('hidden_size', 768), eps=config.get('layer_norm_eps', 1e-12))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


def create_bert_model(variant: str = 'bert_base', config: Optional[Dict[str, Any]] = None) -> BertModel:
    """
    创建BERT模型
    
    Args:
        variant: 模型变体 ('bert_base', 'bert_large', 'bert_tiny')
        config: 模型配置
    
    Returns:
        BERT模型实例
    """
    
    # 预定义配置
    configs = {
        'bert_tiny': {
            'vocab_size': 30522,
            'hidden_size': 128,
            'num_hidden_layers': 2,
            'num_attention_heads': 2,
            'intermediate_size': 512,
            'max_position_embeddings': 512,
        },
        'bert_base': {
            'vocab_size': 30522,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'max_position_embeddings': 512,
        },
        'bert_large': {
            'vocab_size': 30522,
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'max_position_embeddings': 512,
        }
    }
    
    if variant not in configs:
        raise ValueError(f"不支持的BERT变体: {variant}")
    
    # 合并配置
    model_config = configs[variant].copy()
    if config is not None:
        model_config.update(config)
    
    # 设置默认值
    model_config.setdefault('type_vocab_size', 2)
    model_config.setdefault('hidden_dropout_prob', 0.1)
    model_config.setdefault('attention_probs_dropout_prob', 0.1)
    model_config.setdefault('layer_norm_eps', 1e-12)
    model_config.setdefault('initializer_range', 0.02)
    
    return BertModel(model_config)


if __name__ == "__main__":
    # 测试不同变体的BERT模型
    variants = ['bert_tiny', 'bert_base']
    
    print("BERT模型对比:")
    print("-" * 50)
    
    for variant in variants:
        model = create_bert_model(variant)
        params = model.count_parameters()
        print(f"{variant.upper()}: {params:,} 参数")
        
        # 测试前向传播
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            sequence_output, pooled_output, _ = model(input_ids, attention_mask)
        
        print(f"  序列输出形状: {sequence_output.shape}")
        print(f"  池化输出形状: {pooled_output.shape}")
        print()
    
    # 测试掩码语言模型
    print("测试BERT掩码语言模型:")
    config = {'vocab_size': 30522, 'hidden_size': 768, 'num_hidden_layers': 12, 'num_attention_heads': 12}
    mlm_model = BertForMaskedLM(config)
    print(f"MLM模型参数量: {mlm_model.count_parameters():,}")
    
    # 测试序列分类
    print("测试BERT序列分类:")
    config['num_labels'] = 2
    cls_model = BertForSequenceClassification(config)
    print(f"分类模型参数量: {cls_model.count_parameters():,}")
    
    # 前向传播测试
    with torch.no_grad():
        mlm_output = mlm_model(input_ids, attention_mask)
        cls_output = cls_model(input_ids, attention_mask)
    
    print(f"MLM输出形状: {mlm_output['logits'].shape}")
    print(f"分类输出形状: {cls_output['logits'].shape}")