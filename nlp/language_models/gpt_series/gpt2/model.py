"""
GPT-2模型实现

GPT-2是OpenAI在2019年发布的生成式预训练Transformer模型，
是GPT的改进版本，具有更大的规模和更强的文本生成能力。

论文: "Language Models are Unsupervised Multitask Learners" (2019)
作者: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
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


class GPT2Attention(nn.Module):
    """GPT-2多头注意力机制"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.n_head = config.get('n_head', 12)
        self.n_embd = config.get('n_embd', 768)
        self.dropout = config.get('attn_pdrop', 0.1)
        
        assert self.n_embd % self.n_head == 0, "嵌入维度必须能被注意力头数整除"
        
        # key, query, value投影为一个权重矩阵
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        
        # 正则化
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(config.get('resid_pdrop', 0.1))
        
        self.n_ctx = config.get('n_ctx', 1024)
        # 创建因果掩码（下三角矩阵）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.n_ctx, self.n_ctx)).view(1, 1, self.n_ctx, self.n_ctx)
        )
    
    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
              attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算注意力"""
        # q, k, v: [batch, head, seq_len, head_size]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(v.size(-1))
        
        # 获取序列长度
        seq_len = q.size(-2)
        
        # 应用因果掩码
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 应用注意力掩码
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """将隐藏维度分割为多个注意力头"""
        batch_size, seq_len, _ = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head)
        return tensor.permute(0, 2, 1, 3)  # [batch, head, seq_len, head_size]
    
    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """合并多个注意力头"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, head, head_size]
        batch_size, seq_len, _, _ = tensor.size()
        return tensor.view(batch_size, seq_len, self.n_embd)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算查询、键、值
        qkv = self.c_attn(x)
        query, key, value = qkv.split(self.n_embd, dim=2)
        
        # 分割为多头
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # 应用注意力
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        
        # 合并多头
        attn_output = self._merge_heads(attn_output)
        
        # 输出投影
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, attn_weights


class GPT2MLP(nn.Module):
    """GPT-2前馈网络"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        n_embd = config.get('n_embd', 768)
        
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.get('resid_pdrop', 0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    """GPT-2 Transformer块"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        n_embd = config.get('n_embd', 768)
        
        self.ln_1 = nn.LayerNorm(n_embd, eps=config.get('layer_norm_epsilon', 1e-5))
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(n_embd, eps=config.get('layer_norm_epsilon', 1e-5))
        self.mlp = GPT2MLP(config)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 注意力机制
        residual = x
        x = self.ln_1(x)
        attn_output, attn_weights = self.attn(x, attention_mask)
        x = residual + attn_output
        
        # 前馈网络
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x, attn_weights


class GPT2Model(BaseModel):
    """
    GPT-2基础模型
    
    Args:
        config: 模型配置字典
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.config = config
        self.n_embd = config.get('n_embd', 768)
        self.n_layer = config.get('n_layer', 12)
        self.n_head = config.get('n_head', 12)
        self.n_ctx = config.get('n_ctx', 1024)
        self.vocab_size = config.get('vocab_size', 50257)
        
        # Token嵌入
        self.wte = nn.Embedding(self.vocab_size, self.n_embd)
        # 位置嵌入
        self.wpe = nn.Embedding(self.n_ctx, self.n_embd)
        
        # Transformer块
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(self.n_layer)])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(self.n_embd, eps=config.get('layer_norm_epsilon', 1e-5))
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
        
        self.apply(_init_weights)
        
        # 对输出投影应用特殊的缩放初始化
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))
    
    def get_position_ids(self, input_ids: torch.Tensor, past_length: int = 0) -> torch.Tensor:
        """获取位置编码ID"""
        seq_length = input_ids.size(1)
        position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, past_key_values: Optional[Tuple] = None,
                output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            past_key_values: 用于生成的过去键值对
            output_attentions: 是否输出注意力权重
        
        Returns:
            包含last_hidden_state等的字典
        """
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            past_length = 0
            if past_key_values is not None:
                past_length = past_key_values[0][0].size(-2)
            position_ids = self.get_position_ids(input_ids, past_length)
        
        # 嵌入
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # 处理注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 通过Transformer层
        presents = () if past_key_values is not None else None
        all_attentions = () if output_attentions else None
        
        for i, block in enumerate(self.h):
            if past_key_values is not None:
                past_key_value = past_key_values[i]
            else:
                past_key_value = None
            
            hidden_states, attn_weights = block(hidden_states, attention_mask)
            
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)
        
        # 最终层归一化
        hidden_states = self.ln_f(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'attentions': all_attentions
        }


class GPT2LMHeadModel(GPT2Model):
    """用于语言建模的GPT-2"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 语言模型头（与输入嵌入共享权重）
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # 权重共享
        
        self.init_weights()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            labels: [batch_size, seq_len] 语言建模标签
        """
        outputs = super().forward(input_ids, attention_mask, **kwargs)
        hidden_states = outputs['last_hidden_state']
        
        # 计算语言模型logits
        lm_logits = self.lm_head(hidden_states)
        
        result = {
            'logits': lm_logits,
            'hidden_states': hidden_states
        }
        
        if outputs.get('attentions'):
            result['attentions'] = outputs['attentions']
        
        # 计算损失
        if labels is not None:
            # 移位以对齐
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            result['loss'] = loss
        
        return result
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0,
                 do_sample: bool = True, pad_token_id: Optional[int] = None) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入token序列
            max_length: 最大生成长度
            temperature: 温度参数，控制随机性
            top_k: Top-k采样
            top_p: Top-p（nucleus）采样
            do_sample: 是否使用采样
            pad_token_id: 填充token ID
        
        Returns:
            生成的token序列
        """
        self.eval()
        
        # 准备生成
        input_ids = input_ids.clone()
        batch_size = input_ids.size(0)
        
        # 如果没有指定pad_token_id，使用vocab_size-1
        if pad_token_id is None:
            pad_token_id = self.vocab_size - 1
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # 前向传播
                outputs = self.forward(input_ids)
                logits = outputs['logits']
                
                # 获取下一个token的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过阈值的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保留第一个超过阈值的token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove, float('-inf'))
                
                # 采样下一个token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 添加到输入序列
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # 检查是否生成了结束token（如果有的话）
                if hasattr(self, 'eos_token_id') and (next_token == self.eos_token_id).all():
                    break
        
        return input_ids


def create_gpt2_model(variant: str = 'gpt2', config: Optional[Dict[str, Any]] = None) -> GPT2LMHeadModel:
    """
    创建GPT-2模型
    
    Args:
        variant: 模型变体 ('gpt2', 'gpt2_medium', 'gpt2_large', 'gpt2_xl')
        config: 模型配置
    
    Returns:
        GPT2LMHeadModel实例
    """
    
    # 预定义配置
    configs = {
        'gpt2': {  # GPT-2 Small (117M)
            'n_layer': 12,
            'n_head': 12,
            'n_embd': 768,
            'n_ctx': 1024,
            'vocab_size': 50257,
        },
        'gpt2_medium': {  # GPT-2 Medium (345M)
            'n_layer': 24,
            'n_head': 16,
            'n_embd': 1024,
            'n_ctx': 1024,
            'vocab_size': 50257,
        },
        'gpt2_large': {  # GPT-2 Large (762M)
            'n_layer': 36,
            'n_head': 20,
            'n_embd': 1280,
            'n_ctx': 1024,
            'vocab_size': 50257,
        },
        'gpt2_xl': {  # GPT-2 XL (1.5B)
            'n_layer': 48,
            'n_head': 25,
            'n_embd': 1600,
            'n_ctx': 1024,
            'vocab_size': 50257,
        }
    }
    
    if variant not in configs:
        raise ValueError(f"不支持的GPT-2变体: {variant}")
    
    # 合并配置
    model_config = configs[variant].copy()
    if config is not None:
        model_config.update(config)
    
    # 设置默认值
    model_config.setdefault('attn_pdrop', 0.1)
    model_config.setdefault('resid_pdrop', 0.1)
    model_config.setdefault('embd_pdrop', 0.1)
    model_config.setdefault('layer_norm_epsilon', 1e-5)
    
    return GPT2LMHeadModel(model_config)


if __name__ == "__main__":
    # 测试不同变体的GPT-2模型
    variants = ['gpt2', 'gpt2_medium']
    
    print("GPT-2模型对比:")
    print("-" * 50)
    
    for variant in variants:
        model = create_gpt2_model(variant)
        params = model.count_parameters()
        print(f"{variant.upper()}: {params:,} 参数")
        
        # 测试前向传播
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"  Logits形状: {outputs['logits'].shape}")
        print(f"  隐藏状态形状: {outputs['hidden_states'].shape}")
        print()
    
    # 测试生成功能
    print("测试文本生成:")
    model = create_gpt2_model('gpt2')
    
    # 模拟输入prompt
    input_ids = torch.randint(0, 1000, (1, 10))  # 随机初始序列
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=20,
            temperature=0.8,
            top_k=50,
            do_sample=True
        )
    
    print(f"输入长度: {input_ids.size(1)}")
    print(f"生成长度: {generated.size(1)}")
    print(f"生成序列: {generated[0].tolist()}")
    
    # 测试损失计算
    print("\\n测试损失计算:")
    labels = input_ids.clone()
    outputs = model(input_ids, labels=labels)
    print(f"语言建模损失: {outputs['loss'].item():.4f}")