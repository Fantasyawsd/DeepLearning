"""
AlexNet模型实现

AlexNet是由Alex Krizhevsky等人在2012年提出的深度卷积神经网络，
在ImageNet竞赛中取得突破性成果，标志着深度学习时代的开始。

论文: "ImageNet Classification with Deep Convolutional Neural Networks" (2012)
作者: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, Optional

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from shared.base_model import BaseModel


class AlexNet(BaseModel):
    """
    AlexNet网络实现
    
    原始AlexNet包含8层：5个卷积层 + 3个全连接层
    主要创新点：
    - ReLU激活函数
    - Dropout正则化
    - Local Response Normalization (LRN)
    - GPU并行训练
    
    Args:
        num_classes: 分类类别数，默认1000(ImageNet)
        input_channels: 输入通道数，默认3(RGB图像)
        dropout: Dropout概率，默认0.5
        use_lrn: 是否使用局部响应归一化，默认True
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.num_classes = config.get('num_classes', 1000)
        self.input_channels = config.get('input_channels', 3)
        self.dropout = config.get('dropout', 0.5)
        self.use_lrn = config.get('use_lrn', True)
        
        # 特征提取器 (卷积层)
        self.features = nn.Sequential(
            # 第1个卷积层: 224x224x3 -> 55x55x96
            nn.Conv2d(self.input_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2) if self.use_lrn else nn.Identity(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第2个卷积层: 27x27x96 -> 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2) if self.use_lrn else nn.Identity(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第3个卷积层: 13x13x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第4个卷积层: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第5个卷积层: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 自适应平均池化，确保输出尺寸固定为6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 分类器 (全连接层)
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 特征提取
        x = self.features(x)
        # 自适应池化
        x = self.avgpool(x)
        # 展平
        x = torch.flatten(x, 1)
        # 分类
        x = self.classifier(x)
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量（全连接层之前）"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 通过前两个全连接层
        x = self.classifier[0](x)  # Dropout
        x = self.classifier[1](x)  # Linear + ReLU
        x = self.classifier[2](x)  # ReLU
        x = self.classifier[3](x)  # Dropout
        x = self.classifier[4](x)  # Linear + ReLU
        x = self.classifier[5](x)  # ReLU
        
        return x
    
    def get_layer_output(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """获取指定层的输出"""
        if layer_name == 'conv1':
            return self.features[0](x)
        elif layer_name == 'conv2':
            x = self.features[:5](x)
            return self.features[5](x)
        elif layer_name == 'conv3':
            x = self.features[:8](x)
            return self.features[8](x)
        elif layer_name == 'conv4':
            x = self.features[:10](x)
            return self.features[10](x)
        elif layer_name == 'conv5':
            x = self.features[:12](x)
            return self.features[12](x)
        elif layer_name == 'features':
            return self.features(x)
        elif layer_name == 'fc6':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier[0](x)
            return self.classifier[1](x)
        elif layer_name == 'fc7':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier[:4](x)
            return self.classifier[4](x)
        else:
            raise ValueError(f"不支持的层名称: {layer_name}")


class AlexNetBN(BaseModel):
    """
    改进版AlexNet，使用BatchNorm替代LocalResponseNorm
    
    相比原始AlexNet的改进：
    - 用BatchNorm替代LRN，提升训练稳定性
    - 可选的预激活模式
    - 更好的权重初始化
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.num_classes = config.get('num_classes', 1000)
        self.input_channels = config.get('input_channels', 3)
        self.dropout = config.get('dropout', 0.5)
        
        # 特征提取器
        self.features = nn.Sequential(
            # 第1个卷积层
            nn.Conv2d(self.input_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第2个卷积层
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第3个卷积层
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # 第4个卷积层
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # 第5个卷积层
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_alexnet(variant: str = 'alexnet', config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    创建AlexNet模型
    
    Args:
        variant: 模型变体 ('alexnet', 'alexnet_bn')
        config: 模型配置
    
    Returns:
        AlexNet模型实例
    """
    if config is None:
        config = {
            'num_classes': 1000,
            'input_channels': 3,
            'dropout': 0.5
        }
    
    if variant == 'alexnet':
        return AlexNet(config)
    elif variant == 'alexnet_bn':
        return AlexNetBN(config)
    else:
        raise ValueError(f"不支持的AlexNet变体: {variant}")


if __name__ == "__main__":
    # 测试模型
    config = {
        'num_classes': 1000,
        'input_channels': 3,
        'dropout': 0.5
    }
    
    # 测试原始AlexNet
    model = AlexNet(config)
    print(f"AlexNet参数量: {model.count_parameters():,}")
    
    # 测试改进版AlexNet
    model_bn = AlexNetBN(config)
    print(f"AlexNet-BN参数量: {model_bn.count_parameters():,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
        output_bn = model_bn(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出形状(BN): {output_bn.shape}")
    
    # 测试特征提取
    features = model.extract_features(x)
    print(f"特征向量形状: {features.shape}")
    
    # 测试层输出
    conv1_output = model.get_layer_output(x, 'conv1')
    print(f"Conv1输出形状: {conv1_output.shape}")