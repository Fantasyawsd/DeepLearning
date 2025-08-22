"""
VGG模型实现

VGG是由牛津大学视觉几何组(Visual Geometry Group)在2014年提出的深度卷积神经网络。
VGG的核心思想是使用小尺寸卷积核(3x3)构建深度网络。

论文: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014)
作者: Karen Simonyan, Andrew Zisserman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, Optional, List

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from shared.base_model import BaseModel


# VGG网络配置
VGG_CONFIGS = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(BaseModel):
    """
    VGG网络实现
    
    VGG的主要特点：
    - 使用3x3小卷积核
    - 深度网络(11-19层)
    - 统一的网络结构
    - 全连接层作为分类器
    
    Args:
        config: 模型配置字典
            - architecture: VGG架构 ('vgg11', 'vgg13', 'vgg16', 'vgg19')
            - num_classes: 分类类别数
            - input_channels: 输入通道数
            - batch_norm: 是否使用BatchNorm
            - dropout: Dropout概率
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.architecture = config.get('architecture', 'vgg16')
        self.num_classes = config.get('num_classes', 1000)
        self.input_channels = config.get('input_channels', 3)
        self.batch_norm = config.get('batch_norm', False)
        self.dropout = config.get('dropout', 0.5)
        
        # 检查架构是否支持
        if self.architecture not in VGG_CONFIGS:
            raise ValueError(f"不支持的VGG架构: {self.architecture}，支持的架构: {list(VGG_CONFIGS.keys())}")
        
        # 构建特征提取器
        self.features = self._make_layers(VGG_CONFIGS[self.architecture])
        
        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, self.num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layers(self, cfg: List) -> nn.Sequential:
        """根据配置构建卷积层"""
        layers = []
        in_channels = self.input_channels
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量（第一个全连接层的输出）"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 第一个全连接层 + ReLU
        x = self.classifier[0](x)  # Linear
        x = self.classifier[1](x)  # ReLU
        
        return x
    
    def get_layer_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取各层的输出"""
        outputs = {}
        
        # 逐层通过特征提取器
        layer_idx = 0
        conv_idx = 1
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                outputs[f'conv{conv_idx}'] = x
                conv_idx += 1
            elif isinstance(layer, nn.MaxPool2d):
                outputs[f'pool{layer_idx//4 + 1}'] = x
                layer_idx = i
        
        # 池化和展平
        x = self.avgpool(x)
        outputs['avgpool'] = x
        x = torch.flatten(x, 1)
        outputs['flatten'] = x
        
        # 分类器层
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                if i == 0:
                    outputs['fc1'] = x
                elif i == 3:
                    outputs['fc2'] = x
                elif i == 6:
                    outputs['fc3'] = x
        
        outputs['final'] = x
        return outputs


class VGGFeatureExtractor(nn.Module):
    """VGG特征提取器，用于预训练模型的特征提取"""
    
    def __init__(self, vgg_model: VGG, layer_name: str = 'fc1'):
        super().__init__()
        self.vgg = vgg_model
        self.layer_name = layer_name
        
        # 冻结参数
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回指定层的特征"""
        if self.layer_name == 'features':
            return self.vgg.features(x)
        elif self.layer_name == 'fc1':
            return self.vgg.extract_features(x)
        else:
            outputs = self.vgg.get_layer_outputs(x)
            return outputs.get(self.layer_name, outputs['final'])


def create_vgg(
    architecture: str = 'vgg16',
    num_classes: int = 1000,
    batch_norm: bool = False,
    pretrained: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> VGG:
    """
    创建VGG模型
    
    Args:
        architecture: VGG架构 ('vgg11', 'vgg13', 'vgg16', 'vgg19')
        num_classes: 分类类别数
        batch_norm: 是否使用BatchNorm
        pretrained: 是否加载预训练权重(暂不支持)
        config: 模型配置
    
    Returns:
        VGG模型实例
    """
    if config is None:
        config = {
            'architecture': architecture,
            'num_classes': num_classes,
            'input_channels': 3,
            'batch_norm': batch_norm,
            'dropout': 0.5
        }
    
    model = VGG(config)
    
    if pretrained:
        print("警告: 暂不支持预训练权重加载")
    
    return model


# 便捷函数
def vgg11(config: Optional[Dict[str, Any]] = None) -> VGG:
    """VGG-11模型"""
    if config is None:
        config = {'architecture': 'vgg11'}
    else:
        config['architecture'] = 'vgg11'
    return create_vgg(config=config)


def vgg11_bn(config: Optional[Dict[str, Any]] = None) -> VGG:
    """VGG-11模型 (带BatchNorm)"""
    if config is None:
        config = {'architecture': 'vgg11', 'batch_norm': True}
    else:
        config.update({'architecture': 'vgg11', 'batch_norm': True})
    return create_vgg(config=config)


def vgg13(config: Optional[Dict[str, Any]] = None) -> VGG:
    """VGG-13模型"""
    if config is None:
        config = {'architecture': 'vgg13'}
    else:
        config['architecture'] = 'vgg13'
    return create_vgg(config=config)


def vgg13_bn(config: Optional[Dict[str, Any]] = None) -> VGG:
    """VGG-13模型 (带BatchNorm)"""
    if config is None:
        config = {'architecture': 'vgg13', 'batch_norm': True}
    else:
        config.update({'architecture': 'vgg13', 'batch_norm': True})
    return create_vgg(config=config)


def vgg16(config: Optional[Dict[str, Any]] = None) -> VGG:
    """VGG-16模型"""
    if config is None:
        config = {'architecture': 'vgg16'}
    else:
        config['architecture'] = 'vgg16'
    return create_vgg(config=config)


def vgg16_bn(config: Optional[Dict[str, Any]] = None) -> VGG:
    """VGG-16模型 (带BatchNorm)"""
    if config is None:
        config = {'architecture': 'vgg16', 'batch_norm': True}
    else:
        config.update({'architecture': 'vgg16', 'batch_norm': True})
    return create_vgg(config=config)


def vgg19(config: Optional[Dict[str, Any]] = None) -> VGG:
    """VGG-19模型"""
    if config is None:
        config = {'architecture': 'vgg19'}
    else:
        config['architecture'] = 'vgg19'
    return create_vgg(config=config)


def vgg19_bn(config: Optional[Dict[str, Any]] = None) -> VGG:
    """VGG-19模型 (带BatchNorm)"""
    if config is None:
        config = {'architecture': 'vgg19', 'batch_norm': True}
    else:
        config.update({'architecture': 'vgg19', 'batch_norm': True})
    return create_vgg(config=config)


if __name__ == "__main__":
    # 测试不同的VGG模型
    configs = {
        'vgg11': {'architecture': 'vgg11'},
        'vgg13': {'architecture': 'vgg13'},
        'vgg16': {'architecture': 'vgg16'},
        'vgg19': {'architecture': 'vgg19'},
    }
    
    print("VGG模型参数量对比:")
    print("-" * 50)
    
    for arch, config in configs.items():
        config.update({
            'num_classes': 1000,
            'input_channels': 3,
            'batch_norm': False,
            'dropout': 0.5
        })
        
        model = VGG(config)
        params = model.count_parameters()
        print(f"{arch.upper()}: {params:,} 参数")
        
        # 测试前向传播
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"  输出形状: {output.shape}")
        
        # 测试特征提取
        features = model.extract_features(x)
        print(f"  特征形状: {features.shape}")
        print()
    
    # 测试BatchNorm版本
    print("测试VGG16-BN:")
    config_bn = {
        'architecture': 'vgg16',
        'num_classes': 1000,
        'input_channels': 3,
        'batch_norm': True,
        'dropout': 0.5
    }
    
    model_bn = VGG(config_bn)
    print(f"VGG16-BN参数量: {model_bn.count_parameters():,}")
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output_bn = model_bn(x)
        layer_outputs = model_bn.get_layer_outputs(x)
    
    print(f"输出形状: {output_bn.shape}")
    print(f"层输出数量: {len(layer_outputs)}")
    print(f"层名称: {list(layer_outputs.keys())}")