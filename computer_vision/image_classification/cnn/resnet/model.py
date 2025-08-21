"""
ResNet模型实现

ResNet (Residual Network) 是由何恺明等人在2015年提出的深度残差网络，
通过引入残差连接解决了深度网络的梯度消失问题。

论文: "Deep Residual Learning for Image Recognition" (CVPR 2016)
作者: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, Optional, Type, Union, List

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from shared.base_model import BaseModel


class BasicBlock(nn.Module):
    """
    ResNet基础残差块 (用于ResNet18, ResNet34)
    """
    expansion = 1  # 输出通道扩张倍数
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        # 第一个3x3卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 下采样层(用于维度匹配)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    ResNet瓶颈残差块 (用于ResNet50, ResNet101, ResNet152)
    """
    expansion = 4  # 输出通道扩张倍数
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1卷积升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 1x1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 3x3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # 1x1卷积
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(BaseModel):
    """
    ResNet网络实现
    
    支持ResNet18, ResNet34, ResNet50, ResNet101, ResNet152等变体
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.num_classes = config.get('num_classes', 1000)
        self.input_channels = config.get('input_channels', 3)
        
        # 网络架构配置
        block_type = config.get('block_type', 'basic')  # 'basic' or 'bottleneck'
        layers = config.get('layers', [2, 2, 2, 2])  # 每个stage的层数
        
        # 选择残差块类型
        if block_type == 'basic':
            self.block = BasicBlock
        elif block_type == 'bottleneck':
            self.block = Bottleneck
        else:
            raise ValueError(f"不支持的残差块类型: {block_type}")
        
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, self.num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """
        构建残差层
        
        Args:
            out_channels: 输出通道数
            blocks: 残差块数量
            stride: 步长
        """
        downsample = None
        
        # 如果维度不匹配，需要下采样
        if stride != 1 or self.in_channels != out_channels * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.block.expansion)
            )
        
        layers = []
        
        # 第一个残差块(可能需要下采样)
        layers.append(self.block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * self.block.expansion
        
        # 其余残差块
        for _ in range(1, blocks):
            layers.append(self.block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类器
        x = self.fc(x)
        
        return x
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ResNet变体配置
RESNET_CONFIGS = {
    'resnet18': {
        'block_type': 'basic',
        'layers': [2, 2, 2, 2],
        'description': 'ResNet-18: 18层残差网络，使用BasicBlock'
    },
    'resnet34': {
        'block_type': 'basic', 
        'layers': [3, 4, 6, 3],
        'description': 'ResNet-34: 34层残差网络，使用BasicBlock'
    },
    'resnet50': {
        'block_type': 'bottleneck',
        'layers': [3, 4, 6, 3],
        'description': 'ResNet-50: 50层残差网络，使用Bottleneck'
    },
    'resnet101': {
        'block_type': 'bottleneck',
        'layers': [3, 4, 23, 3],
        'description': 'ResNet-101: 101层残差网络，使用Bottleneck'
    },
    'resnet152': {
        'block_type': 'bottleneck',
        'layers': [3, 8, 36, 3],
        'description': 'ResNet-152: 152层残差网络，使用Bottleneck'
    }
}


def create_resnet(variant: str, config: Optional[Dict[str, Any]] = None) -> ResNet:
    """
    创建ResNet模型的便捷函数
    
    Args:
        variant: ResNet变体 ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        config: 模型配置
        
    Returns:
        ResNet模型实例
    """
    if variant not in RESNET_CONFIGS:
        raise ValueError(f"不支持的ResNet变体: {variant}. 可用: {list(RESNET_CONFIGS.keys())}")
    
    if config is None:
        config = {}
    
    # 合并预设配置
    resnet_config = RESNET_CONFIGS[variant].copy()
    resnet_config.update(config)
    
    return ResNet(resnet_config)


class PreActResNet(BaseModel):
    """
    Pre-activation ResNet实现
    
    在论文 "Identity Mappings in Deep Residual Networks" 中提出，
    将BatchNorm和ReLU移到卷积之前，获得更好的梯度流。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_classes = config.get('num_classes', 10)
        self.input_channels = config.get('input_channels', 3)
        layers = config.get('layers', [2, 2, 2, 2])
        
        self.in_channels = 64
        
        # 初始卷积(不使用BatchNorm和ReLU)
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        
        # 残差层
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # 最终BatchNorm
        self.bn = nn.BatchNorm2d(512)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        """构建Pre-activation残差层"""
        layers = []
        
        for i in range(blocks):
            if i == 0:
                layers.append(PreActBlock(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            else:
                layers.append(PreActBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.bn(x)
        x = F.relu(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PreActBlock(nn.Module):
    """Pre-activation残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        
        # 如果维度不匹配，使用1x1卷积
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        
        return out + shortcut


# 数据集特定配置
DATASET_CONFIGS = {
    'imagenet': {
        'num_classes': 1000,
        'input_channels': 3,
        'input_size': 224
    },
    'cifar10': {
        'num_classes': 10,
        'input_channels': 3,
        'input_size': 32
    },
    'cifar100': {
        'num_classes': 100,
        'input_channels': 3,
        'input_size': 32
    }
}


def get_resnet_config(dataset: str, variant: str) -> Dict[str, Any]:
    """
    获取特定数据集和变体的ResNet配置
    
    Args:
        dataset: 数据集名称
        variant: ResNet变体
        
    Returns:
        配置字典
    """
    config = RESNET_CONFIGS[variant].copy()
    config.update(DATASET_CONFIGS.get(dataset, DATASET_CONFIGS['imagenet']))
    return config