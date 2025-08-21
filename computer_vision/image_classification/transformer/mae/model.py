"""
MAE (Masked Autoencoder) 模型实现

基于论文 "Masked Autoencoders Are Scalable Vision Learners" by He et al.
https://arxiv.org/abs/2111.06377

MAE是一种自监督学习的视觉Transformer模型，通过随机掩码图像块并重建它们来学习视觉表示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from typing import Tuple, Optional
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from models.base import BaseModel


class PatchEmbed(nn.Module):
    """图像到补丁嵌入 (Image to Patch Embedding)"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入图像尺寸 ({H}*{W}) 与模型尺寸 ({self.img_size}*{self.img_size}) 不匹配"
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力模块 (Multi-Head Self Attention)"""
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """多层感知机 (Multi-Layer Perceptron)"""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, 
                 act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块 (Transformer Block)"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False,
                 drop: float = 0., attn_drop: float = 0., norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """MAE编码器 (MAE Encoder)"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化权重"""
        # 初始化position embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        随机掩码序列
        
        Args:
            x: [N, L, D], sequence
            mask_ratio: 掩码比例
            
        Returns:
            x_masked: [N, L_visible, D], 可见tokens
            mask: [N, L], 0表示保留, 1表示移除  
            ids_restore: [N, L], 用于恢复原始顺序
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # 排序噪声获得shuffle和unshuffle
        ids_shuffle = torch.argsort(noise, dim=1)  # 升序排列: 小的保留, 大的移除
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留第一个子集
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # 生成二进制掩码: 0表示保留, 1表示移除
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle获得二进制掩码
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 嵌入patches
        x = self.patch_embed(x)
        
        # 添加位置编码w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # 掩码: 长度 -> 长度 * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # 添加cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 应用Transformer块
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore


class MAEDecoder(nn.Module):
    """MAE解码器 (MAE Decoder)"""
    
    def __init__(self, num_patches: int, patch_size: int, in_chans: int = 3, embed_dim: int = 512,
                 decoder_embed_dim: int = 512, decoder_depth: int = 8, decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.0, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
    
    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        # 嵌入tokens
        x = self.decoder_embed(x)
        
        # 添加mask tokens到序列
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # 不要cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # 添加cls token
        
        # 添加位置编码
        x = x + self.decoder_pos_embed
        
        # 应用Transformer块
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # 预测器
        x = self.decoder_pred(x)
        
        # 移除cls token
        x = x[:, 1:, :]
        
        return x


class MAE(BaseModel):
    """
    MAE (Masked Autoencoder) 模型
    
    MAE是一种自监督学习的视觉Transformer，通过随机掩码图像块并重建它们来学习视觉表示。
    
    Args:
        img_size: 输入图像尺寸
        patch_size: 补丁尺寸  
        in_chans: 输入通道数
        embed_dim: 编码器嵌入维度
        depth: 编码器深度
        num_heads: 编码器注意力头数
        decoder_embed_dim: 解码器嵌入维度
        decoder_depth: 解码器深度
        decoder_num_heads: 解码器注意力头数
        mlp_ratio: MLP隐藏层维度比例
        norm_pix_loss: 是否标准化像素损失
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 从配置中获取参数
        img_size = config.get('img_size', 224)
        patch_size = config.get('patch_size', 16)
        in_chans = config.get('in_chans', 3)
        embed_dim = config.get('embed_dim', 1024)
        depth = config.get('depth', 24)
        num_heads = config.get('num_heads', 16)
        decoder_embed_dim = config.get('decoder_embed_dim', 512)
        decoder_depth = config.get('decoder_depth', 8)
        decoder_num_heads = config.get('decoder_num_heads', 16)
        mlp_ratio = config.get('mlp_ratio', 4.0)
        norm_layer = nn.LayerNorm
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = config.get('norm_pix_loss', False)
        
        # 编码器
        self.encoder = MAEEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer
        )
        
        # 解码器
        self.decoder = MAEDecoder(
            num_patches=self.encoder.patch_embed.num_patches,
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        将图像转换为patches
        
        Args:
            imgs: (N, 3, H, W)
        Returns:
            x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将patches转换为图像
        
        Args:
            x: (N, L, patch_size**2 *3)
        Returns:
            imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs
    
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算重建损失
        
        Args:
            imgs: 原始图像
            pred: 预测结果
            mask: 掩码
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], 每个patch的均值损失
        
        loss = (loss * mask).sum() / mask.sum()  # 只计算被移除patches的均值损失
        return loss
    
    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            imgs: 输入图像 (N, C, H, W)
            mask_ratio: 掩码比例
            
        Returns:
            loss: 重建损失
            pred: 预测结果
            mask: 掩码
        """
        latent, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred = self.decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)