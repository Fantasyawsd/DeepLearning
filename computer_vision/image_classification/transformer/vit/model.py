"""
ViT (Vision Transformer) 模型实现

Vision Transformer是Google在2020年提出的将Transformer架构应用于计算机视觉的开创性工作。

论文: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)
作者: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import Dict, Any, Optional, Tuple
from einops import rearrange, repeat

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from shared.base_model import BaseModel


class PatchEmbedding(nn.Module):
    """
    图像补丁嵌入层
    
    将图像分割为固定大小的补丁，并将每个补丁映射为嵌入向量
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层实现补丁嵌入
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            (batch_size, num_patches, embed_dim)
        """
        # 卷积 + 重塑
        x = self.projection(x)  # (B, embed_dim, H//P, W//P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV线性变换
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        B, N, C = x.shape
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.0, activation: nn.Module = nn.GELU):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = activation()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多头自注意力 + 残差连接
        x = x + self.attn(self.norm1(x))
        
        # 前馈网络 + 残差连接
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(BaseModel):
    """
    Vision Transformer (ViT) 实现
    
    将图像分割为补丁序列，使用Transformer处理图像分类任务
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.img_size = config.get('img_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.in_channels = config.get('in_channels', 3)
        self.num_classes = config.get('num_classes', 1000)
        
        self.embed_dim = config.get('embed_dim', 768)
        self.depth = config.get('depth', 12)
        self.num_heads = config.get('num_heads', 12)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        
        self.dropout = config.get('dropout', 0.1)
        self.attn_dropout = config.get('attn_dropout', 0.0)
        
        # 补丁嵌入
        self.patch_embed = PatchEmbedding(
            self.img_size, self.patch_size, self.in_channels, self.embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_dropout = nn.Dropout(self.dropout)
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.embed_dim, self.num_heads, self.mlp_ratio,
                self.dropout, self.attn_dropout
            )
            for _ in range(self.depth)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        # 位置编码使用截断正态分布初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 其他层使用标准初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, channels, height, width)
        Returns:
            分类logits (batch_size, num_classes)
        """
        B = x.shape[0]
        
        # 补丁嵌入
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer编码器
        for block in self.blocks:
            x = block(x)
        
        # 归一化
        x = self.norm(x)
        
        # 分类(使用CLS token)
        cls_token_final = x[:, 0]  # (B, embed_dim)
        x = self.head(cls_token_final)
        
        return x
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        获取指定层的注意力图
        
        Args:
            x: 输入图像
            layer_idx: 层索引，-1表示最后一层
        """
        B = x.shape[0]
        
        # 前向传播到指定层
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        if layer_idx == -1:
            layer_idx = len(self.blocks) - 1
        
        for i, block in enumerate(self.blocks):
            if i < layer_idx:
                x = block(x)
            elif i == layer_idx:
                # 在目标层提取注意力权重
                x = x + block.attn(block.norm1(x))
                # 这里需要修改MultiHeadSelfAttention来返回注意力权重
                break
        
        return x
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ViT模型变体配置
VIT_CONFIGS = {
    'vit_tiny_patch16_224': {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4.0,
        'description': 'ViT-Tiny: 轻量级Vision Transformer'
    },
    'vit_small_patch16_224': {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'description': 'ViT-Small: 小型Vision Transformer'
    },
    'vit_base_patch16_224': {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'description': 'ViT-Base: 基础Vision Transformer'
    },
    'vit_large_patch16_224': {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'description': 'ViT-Large: 大型Vision Transformer'
    },
    'vit_huge_patch14_224': {
        'img_size': 224,
        'patch_size': 14,
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'description': 'ViT-Huge: 巨型Vision Transformer'
    }
}


def create_vit(variant: str, config: Optional[Dict[str, Any]] = None) -> VisionTransformer:
    """
    创建ViT模型的便捷函数
    
    Args:
        variant: ViT变体名称
        config: 额外配置
    
    Returns:
        ViT模型实例
    """
    if variant not in VIT_CONFIGS:
        raise ValueError(f"不支持的ViT变体: {variant}. 可用: {list(VIT_CONFIGS.keys())}")
    
    model_config = VIT_CONFIGS[variant].copy()
    if config:
        model_config.update(config)
    
    return VisionTransformer(model_config)


class DeiT(VisionTransformer):
    """
    DeiT (Data-efficient Image Transformers) 实现
    
    基于ViT但使用知识蒸馏训练，在较小数据集上也能获得良好性能
    
    论文: "Training data-efficient image transformers & distillation through attention" (ICML 2021)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # DeiT特有的蒸馏token
        self.use_distillation = config.get('use_distillation', False)
        if self.use_distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            # 蒸馏分类头
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes)
            
            # 重新初始化位置编码(现在有额外的蒸馏token)
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
            
            nn.init.trunc_normal_(self.dist_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DeiT前向传播"""
        B = x.shape[0]
        
        # 补丁嵌入
        x = self.patch_embed(x)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 如果使用蒸馏，添加蒸馏token
        if self.use_distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([x[:, :1], dist_tokens, x[:, 1:]], dim=1)
        
        # 位置编码
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer编码器
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 分类
        if self.use_distillation:
            cls_output = self.head(x[:, 0])
            dist_output = self.head_dist(x[:, 1])
            return cls_output, dist_output
        else:
            return self.head(x[:, 0])


def create_deit(variant: str, use_distillation: bool = False, 
                config: Optional[Dict[str, Any]] = None) -> DeiT:
    """创建DeiT模型"""
    if variant not in VIT_CONFIGS:
        raise ValueError(f"不支持的DeiT变体: {variant}")
    
    model_config = VIT_CONFIGS[variant].copy()
    model_config['use_distillation'] = use_distillation
    
    if config:
        model_config.update(config)
    
    return DeiT(model_config)


# 数据集配置
DATASET_VIT_CONFIGS = {
    'imagenet': {
        'num_classes': 1000,
        'in_channels': 3,
        'img_size': 224
    },
    'cifar10': {
        'num_classes': 10,
        'in_channels': 3,
        'img_size': 32
    },
    'cifar100': {
        'num_classes': 100,
        'in_channels': 3,
        'img_size': 32
    }
}


def get_vit_config(variant: str, dataset: str) -> Dict[str, Any]:
    """获取特定数据集的ViT配置"""
    config = VIT_CONFIGS[variant].copy()
    config.update(DATASET_VIT_CONFIGS.get(dataset, DATASET_VIT_CONFIGS['imagenet']))
    return config