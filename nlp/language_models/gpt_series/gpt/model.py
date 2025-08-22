"""
GPT (Generative Pre-trained Transformer) 模型实现

GPT是OpenAI开发的自回归语言模型，使用Transformer解码器架构。

论文: "Improving Language Understanding by Generative Pre-Training" (2018)
作者: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import Dict, Any, Optional, Tuple

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from shared.base_model import BaseModel


class MultiHeadAttention(nn.Module):
    """多头自注意力机制 (带因果掩码)"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # 注册因果掩码 (下三角矩阵)
        self.register_buffer("causal_mask", torch.tril(torch.ones(1024, 1024)))
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            attention_mask: (batch_size, seq_len) 可选的attention掩码
        """
        B, T, C = x.shape
        
        # 计算QKV
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        
        # 应用因果掩码
        causal_mask = self.causal_mask[:T, :T]
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        # 应用额外的attention掩码(如padding掩码)
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            attention_mask = attention_mask.view(B, 1, 1, T)
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax和dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        out = attn @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, embed_dim)
        
        # 输出投影
        out = self.proj(out)
        
        return out


class MLP(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, embed_dim: int, intermediate_size: Optional[int] = None, 
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = 4 * embed_dim
        
        self.fc1 = nn.Linear(embed_dim, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # Swish = SiLU
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):
    """GPT Transformer块"""
    
    def __init__(self, embed_dim: int, num_heads: int, intermediate_size: Optional[int] = None,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = MLP(embed_dim, intermediate_size, dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LayerNorm + 残差连接
        residual = x
        x = self.ln1(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        
        # Pre-LayerNorm + 残差连接
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class GPT(BaseModel):
    """
    GPT (Generative Pre-trained Transformer) 模型
    
    自回归语言模型，使用Transformer解码器架构
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.vocab_size = config.get('vocab_size', 50257)  # GPT-2默认词汇表大小
        self.max_seq_len = config.get('max_seq_len', 1024)
        self.embed_dim = config.get('embed_dim', 768)
        self.num_layers = config.get('num_layers', 12)
        self.num_heads = config.get('num_heads', 12)
        self.intermediate_size = config.get('intermediate_size', None)
        self.dropout = config.get('dropout', 0.1)
        self.layer_norm_eps = config.get('layer_norm_eps', 1e-5)
        
        # Token嵌入
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # 位置嵌入
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Transformer层
        self.blocks = nn.ModuleList([
            GPTBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                dropout=self.dropout,
                layer_norm_eps=self.layer_norm_eps
            )
            for _ in range(self.num_layers)
        ])
        
        # 最终LayerNorm
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        
        # 语言模型头
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        
        # 权重共享 (Token embedding和LM head)
        if config.get('tie_word_embeddings', True):
            self.lm_head.weight = self.token_embedding.weight
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        # 使用正态分布初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        
        # 特殊初始化输出投影层
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))
            nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple] = None, use_cache: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            past_key_values: 缓存的k,v值 (用于生成)
            use_cache: 是否返回缓存的k,v值
            
        Returns:
            包含logits和可选缓存的字典
        """
        batch_size, seq_len = input_ids.shape
        
        # 检查序列长度
        if seq_len > self.max_seq_len:
            raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_seq_len}")
        
        # 位置编码
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout_layer(hidden_states)
        
        # Transformer层
        presents = () if use_cache else None
        for i, block in enumerate(self.blocks):
            # TODO: 实现KV缓存用于高效生成
            hidden_states = block(hidden_states, attention_mask)
        
        # 最终LayerNorm
        hidden_states = self.ln_f(hidden_states)
        
        # 语言模型头
        logits = self.lm_head(hidden_states)
        
        output = {'logits': logits}
        if use_cache:
            output['past_key_values'] = presents
        
        return output
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, 
                temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None, do_sample: bool = True,
                pad_token_id: Optional[int] = None) -> torch.Tensor:
        """
        文本生成
        
        Args:
            input_ids: 输入序列 (batch_size, seq_len)
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-k采样
            top_p: Top-p (nucleus) 采样
            do_sample: 是否使用采样
            pad_token_id: Padding token id
            
        Returns:
            生成的序列 (batch_size, max_length)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 复制输入作为生成的起始
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # 前向传播
                outputs = self.forward(generated)
                logits = outputs['logits']
                
                # 获取最后一个时间步的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k过滤
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p过滤
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率大于top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样或贪心解码
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
                
                # 检查是否所有序列都生成了结束符
                if pad_token_id is not None and (next_token == pad_token_id).all():
                    break
        
        return generated
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# GPT模型变体配置
GPT_CONFIGS = {
    'gpt-small': {
        'vocab_size': 50257,
        'max_seq_len': 1024,
        'embed_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'intermediate_size': 3072,
        'description': 'GPT Small: 117M参数'
    },
    'gpt-medium': {
        'vocab_size': 50257,
        'max_seq_len': 1024,
        'embed_dim': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'intermediate_size': 4096,
        'description': 'GPT Medium: 345M参数'
    },
    'gpt-large': {
        'vocab_size': 50257,
        'max_seq_len': 1024,
        'embed_dim': 1280,
        'num_layers': 36,
        'num_heads': 20,
        'intermediate_size': 5120,
        'description': 'GPT Large: 762M参数'
    },
    'gpt-xl': {
        'vocab_size': 50257,
        'max_seq_len': 1024,
        'embed_dim': 1600,
        'num_layers': 48,
        'num_heads': 25,
        'intermediate_size': 6400,
        'description': 'GPT XL: 1.5B参数'
    }
}


def create_gpt(variant: str, config: Optional[Dict[str, Any]] = None) -> GPT:
    """
    创建GPT模型的便捷函数
    
    Args:
        variant: GPT变体 ('gpt-small', 'gpt-medium', 'gpt-large', 'gpt-xl')
        config: 额外配置
        
    Returns:
        GPT模型实例
    """
    if variant not in GPT_CONFIGS:
        raise ValueError(f"不支持的GPT变体: {variant}. 可用: {list(GPT_CONFIGS.keys())}")
    
    model_config = GPT_CONFIGS[variant].copy()
    if config:
        model_config.update(config)
    
    return GPT(model_config)


class GPTForSequenceClassification(GPT):
    """
    用于序列分类的GPT模型
    
    在GPT基础上添加分类头
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_labels = config.get('num_labels', 2)
        
        # 分类头
        self.classifier = nn.Linear(self.embed_dim, self.num_labels)
        
        # 初始化分类头权重
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """分类任务前向传播"""
        # 获取基础模型输出
        outputs = super().forward(input_ids, attention_mask)
        hidden_states = self.ln_f(outputs['logits'])  # 使用最后一层的输出
        
        # 获取序列表示 (使用最后一个非padding token的hidden state)
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            sequence_repr = hidden_states[range(batch_size), sequence_lengths]
        else:
            sequence_repr = hidden_states[:, -1]  # 使用最后一个token
        
        # 分类
        logits = self.classifier(sequence_repr)
        
        output = {'logits': logits}
        
        # 计算损失
        if labels is not None:
            if self.num_labels == 1:
                # 回归任务
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # 分类任务
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output['loss'] = loss
        
        return output


def create_gpt_classifier(variant: str, num_labels: int, 
                         config: Optional[Dict[str, Any]] = None) -> GPTForSequenceClassification:
    """创建GPT分类模型"""
    model_config = GPT_CONFIGS[variant].copy()
    model_config['num_labels'] = num_labels
    
    if config:
        model_config.update(config)
    
    return GPTForSequenceClassification(model_config)