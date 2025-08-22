"""
Swin Transformer模型实现

Swin Transformer是微软在2021年提出的分层视觉Transformer，
通过滑动窗口注意力机制实现线性计算复杂度。

论文: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (2021)
作者: Ze Liu, Yutong Lin, Yue Cao, et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np
import sys
import os

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from shared.base_model import BaseModel


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图分割为不重叠的窗口
    
    Args:
        x: (B, H, W, C) 输入特征图
        window_size: 窗口大小
    
    Returns:
        windows: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    将窗口合并回特征图
    
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size: 窗口大小
        H: 原始高度
        W: 原始宽度
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """窗口多头自注意力模块"""
    
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, 
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 定义相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # 获取窗口内每个token的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 偏移到0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入特征，形状为 (B*num_windows, N, C)
            mask: (0/-inf) mask，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 (B_, num_heads, N, head_dim)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, 
                 window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0., norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            # 如果窗口大小大于输入分辨率，不进行窗口分割
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        if self.shift_size > 0:
            # 计算SW-MSA的注意力掩码
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征长度与分辨率不匹配"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 分割窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        
        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        
        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class PatchMerging(nn.Module):
    """下采样层，合并相邻的patches"""
    
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征长度与分辨率不匹配"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) 必须是偶数"
        
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class BasicLayer(nn.Module):
    """Swin Transformer的基本层"""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0., norm_layer: nn.Module = nn.LayerNorm,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # 构建blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                               num_heads=num_heads, window_size=window_size,
                               shift_size=0 if (i % 2 == 0) else window_size // 2,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer)
            for i in range(depth)])
        
        # 下采样层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """图像到补丁嵌入"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96, 
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # FIXME 可能需要处理输入尺寸不匹配的情况
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像尺寸 ({H}*{W}) 与模型尺寸 ({self.img_size[0]}*{self.img_size[1]}) 不匹配"
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP层"""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, drop: float = 0.):
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


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob: Optional[float] = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class SwinTransformer(BaseModel):
    """
    Swin Transformer主体模型
    
    Args:
        config: 模型配置字典
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.img_size = config.get('img_size', 224)
        self.patch_size = config.get('patch_size', 4)
        self.in_chans = config.get('in_chans', 3)
        self.num_classes = config.get('num_classes', 1000)
        self.embed_dim = config.get('embed_dim', 96)
        self.depths = config.get('depths', [2, 2, 6, 2])
        self.num_heads = config.get('num_heads', [3, 6, 12, 24])
        self.window_size = config.get('window_size', 7)
        self.mlp_ratio = config.get('mlp_ratio', 4.)
        self.qkv_bias = config.get('qkv_bias', True)
        self.drop_rate = config.get('drop_rate', 0.)
        self.attn_drop_rate = config.get('attn_drop_rate', 0.)
        self.drop_path_rate = config.get('drop_path_rate', 0.1)
        
        norm_layer = nn.LayerNorm
        
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        
        # 分割图像为不重叠的patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans,
            embed_dim=self.embed_dim, norm_layer=norm_layer)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # 绝对位置嵌入
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        
        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        
        # 构建层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                              input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                              patches_resolution[1] // (2 ** i_layer)),
                              depth=self.depths[i_layer],
                              num_heads=self.num_heads[i_layer],
                              window_size=self.window_size,
                              mlp_ratio=self.mlp_ratio,
                              qkv_bias=self.qkv_bias,
                              drop=self.drop_rate,
                              attn_drop=self.attn_drop_rate,
                              drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                              norm_layer=norm_layer,
                              downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取"""
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.forward_features(x)
        x = self.head(x)
        return x


def create_swin_transformer(variant: str = 'swin_tiny', config: Optional[Dict[str, Any]] = None) -> SwinTransformer:
    """
    创建Swin Transformer模型
    
    Args:
        variant: 模型变体 ('swin_tiny', 'swin_small', 'swin_base', 'swin_large')
        config: 模型配置
    
    Returns:
        SwinTransformer模型实例
    """
    
    # 预定义配置
    configs = {
        'swin_tiny': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
        },
        'swin_small': {
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
        },
        'swin_base': {
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'window_size': 7,
        },
        'swin_large': {
            'embed_dim': 192,
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
            'window_size': 7,
        }
    }
    
    if variant not in configs:
        raise ValueError(f"不支持的Swin Transformer变体: {variant}")
    
    # 合并配置
    model_config = configs[variant].copy()
    if config is not None:
        model_config.update(config)
    
    # 设置默认值
    model_config.setdefault('img_size', 224)
    model_config.setdefault('patch_size', 4)
    model_config.setdefault('in_chans', 3)
    model_config.setdefault('num_classes', 1000)
    model_config.setdefault('mlp_ratio', 4.)
    model_config.setdefault('qkv_bias', True)
    model_config.setdefault('drop_rate', 0.)
    model_config.setdefault('attn_drop_rate', 0.)
    model_config.setdefault('drop_path_rate', 0.1)
    
    return SwinTransformer(model_config)


if __name__ == "__main__":
    # 测试不同变体的Swin Transformer
    variants = ['swin_tiny', 'swin_small', 'swin_base']
    
    print("Swin Transformer模型对比:")
    print("-" * 60)
    
    for variant in variants:
        model = create_swin_transformer(variant)
        params = model.count_parameters()
        print(f"{variant.upper()}: {params:,} 参数")
        
        # 测试前向传播
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
            features = model.forward_features(x)
        
        print(f"  输出形状: {output.shape}")
        print(f"  特征形状: {features.shape}")
        print()
    
    # 测试自定义配置
    print("测试自定义配置:")
    custom_config = {
        'num_classes': 10,  # CIFAR-10
        'img_size': 32,     # CIFAR-10图像尺寸
        'patch_size': 2,    # 小patch适应小图像
    }
    
    model_custom = create_swin_transformer('swin_tiny', custom_config)
    print(f"自定义模型参数量: {model_custom.count_parameters():,}")
    
    x_small = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output_custom = model_custom(x_small)
    
    print(f"自定义模型输出形状: {output_custom.shape}")