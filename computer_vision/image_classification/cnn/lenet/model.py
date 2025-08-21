"""
LeNet模型实现

LeNet是由Yann LeCun在1998年提出的经典卷积神经网络，是深度学习历史上的里程碑之一。
主要用于手写数字识别任务。

论文: "Gradient-based learning applied to document recognition" (1998)
作者: Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, Optional

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from models.base import BaseModel


class LeNet5(BaseModel):
    """
    LeNet-5网络实现
    
    经典的卷积神经网络架构，包含：
    - 2个卷积层
    - 2个子采样层(平均池化)
    - 3个全连接层
    
    Args:
        num_classes: 分类类别数，默认10(适用于MNIST/CIFAR-10)
        input_channels: 输入通道数，默认1(灰度图像)
        input_size: 输入图像尺寸，默认32
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.num_classes = config.get('num_classes', 10)
        self.input_channels = config.get('input_channels', 1)
        self.input_size = config.get('input_size', 32)
        
        # 第一个卷积层: 输入32x32, 输出28x28
        self.conv1 = nn.Conv2d(self.input_channels, 6, kernel_size=5)
        
        # 第一个池化层: 输出14x14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积层: 输入14x14, 输出10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # 第二个池化层: 输出5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入尺寸
        self.fc_input_size = self._calculate_fc_input_size()
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _calculate_fc_input_size(self):
        """计算全连接层输入尺寸"""
        # 模拟前向传播计算尺寸
        x = torch.randn(1, self.input_channels, self.input_size, self.input_size)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.numel()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, channels, height, width)
            
        Returns:
            输出logits (batch_size, num_classes)
        """
        # 第一个卷积+池化块
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二个卷积+池化块
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LeNetVariant(BaseModel):
    """
    LeNet变体实现
    
    在原始LeNet基础上的改进版本：
    - 使用ReLU激活函数替代Sigmoid
    - 使用MaxPooling替代AveragePooling
    - 添加Batch Normalization
    - 添加Dropout防止过拟合
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.num_classes = config.get('num_classes', 10)
        self.input_channels = config.get('input_channels', 1)
        self.input_size = config.get('input_size', 32)
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(self.input_channels, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6) if self.use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16) if self.use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块(新增)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32) if self.use_batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入尺寸
        self.fc_input_size = self._calculate_fc_input_size()
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(128, self.num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _calculate_fc_input_size(self):
        """计算全连接层输入尺寸"""
        x = torch.randn(1, self.input_channels, self.input_size, self.input_size)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x.numel()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 卷积块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 卷积块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 卷积块3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_lenet(variant: str = 'original', config: Optional[Dict[str, Any]] = None) -> BaseModel:
    """
    创建LeNet模型的便捷函数
    
    Args:
        variant: 模型变体 ('original', 'improved')
        config: 模型配置
        
    Returns:
        LeNet模型实例
    """
    if config is None:
        config = {}
    
    if variant == 'original':
        return LeNet5(config)
    elif variant == 'improved':
        return LeNetVariant(config)
    else:
        raise ValueError(f"不支持的LeNet变体: {variant}")


# 模型配置预设
LENET_CONFIGS = {
    'mnist': {
        'num_classes': 10,
        'input_channels': 1,
        'input_size': 28,
        'use_batch_norm': False,
        'dropout_rate': 0.0
    },
    'cifar10': {
        'num_classes': 10,
        'input_channels': 3,
        'input_size': 32,
        'use_batch_norm': True,
        'dropout_rate': 0.2
    },
    'fashion_mnist': {
        'num_classes': 10,
        'input_channels': 1,
        'input_size': 28,
        'use_batch_norm': True,
        'dropout_rate': 0.1
    }
}


def get_lenet_config(dataset: str) -> Dict[str, Any]:
    """
    获取特定数据集的LeNet配置
    
    Args:
        dataset: 数据集名称
        
    Returns:
        配置字典
    """
    return LENET_CONFIGS.get(dataset, LENET_CONFIGS['cifar10'])