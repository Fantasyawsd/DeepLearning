"""
Masked Autoencoder (MAE) implementation in PyTorch.

Based on "Masked Autoencoders Are Scalable Vision Learners" by He et al.
https://arxiv.org/abs/2111.06377
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from typing import Tuple, Optional

from .base import BaseModel


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention module."""
    
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
    """MLP with GELU activation."""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
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
    """Transformer Block with Multi-Head Attention and MLP."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False,
                 drop: float = 0., attn_drop: float = 0., norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAE(BaseModel):
    """
    Masked Autoencoder (MAE) for self-supervised learning.
    
    Args:
        img_size: Input image size
        patch_size: Patch size for patch embedding
        in_chans: Number of input channels
        embed_dim: Embedding dimension
        encoder_depth: Number of encoder transformer blocks
        encoder_num_heads: Number of attention heads in encoder
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of decoder transformer blocks
        decoder_num_heads: Number of attention heads in decoder
        mlp_ratio: MLP ratio for transformer blocks
        mask_ratio: Masking ratio for training
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Config parameters
        self.img_size = config.get('img_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.in_chans = config.get('in_chans', 3)
        self.embed_dim = config.get('embed_dim', 768)
        self.encoder_depth = config.get('encoder_depth', 12)
        self.encoder_num_heads = config.get('encoder_num_heads', 12)
        self.decoder_embed_dim = config.get('decoder_embed_dim', 512)
        self.decoder_depth = config.get('decoder_depth', 8)
        self.decoder_num_heads = config.get('decoder_num_heads', 16)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.mask_ratio = config.get('mask_ratio', 0.75)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_chans, self.embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.encoder_num_heads, self.mlp_ratio)
            for _ in range(self.encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(self.embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.decoder_embed_dim), requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio)
            for _ in range(self.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.in_chans)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        pos_embed = self.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = self.get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # Initialize weights
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
        """Generate 2D sin-cos position embedding."""
        import numpy as np
        
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = MAE.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed
    
    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        import numpy as np
        
        assert embed_dim % 2 == 0
        
        # use half of dimensions to encode grid_h
        emb_h = MAE.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = MAE.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb
    
    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        import numpy as np
        
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb
    
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform random masking."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder."""
        # Embed patches
        x = self.patch_embed(x)
        
        # Add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder."""
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute forward loss."""
        target = self.patchify(imgs)
        
        # Normalize targets
        if self.config.get('norm_pix_loss', False):
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches."""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images."""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward(self, imgs: torch.Tensor, mask_ratio: Optional[float] = None) -> dict:
        """Forward pass."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        
        return {
            'loss': loss,
            'pred': pred,
            'mask': mask,
            'latent': latent
        }