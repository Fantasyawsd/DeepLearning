"""
LeNet预训练模型加载模块

虽然LeNet是经典的轻量级模型，通常不需要预训练权重，
但本模块提供了从不同来源加载预训练权重的功能
"""

import torch
import torch.nn as nn
import os
import sys
from typing import Dict, Any, Optional

# 添加路径以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

from model import LeNet5


def load_pretrained_lenet(
    variant: str = 'lenet5_mnist',
    config: Optional[Dict[str, Any]] = None,
    pretrained_path: Optional[str] = None
) -> LeNet5:
    """
    加载预训练的LeNet模型
    
    Args:
        variant: 模型变体 ('lenet5_mnist', 'lenet5_cifar10', 'lenet5_cifar100')
        config: 模型配置，如果为None则使用默认配置
        pretrained_path: 预训练权重路径，如果为None则使用默认权重
    
    Returns:
        加载了预训练权重的LeNet5模型
    """
    
    # 默认配置
    if config is None:
        config = get_default_config(variant)
    
    # 创建模型
    model = LeNet5(config)
    
    # 加载预训练权重
    if pretrained_path is not None and os.path.exists(pretrained_path):
        print(f"从 {pretrained_path} 加载预训练权重")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 提取模型权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        print("预训练权重加载成功")
    else:
        print(f"未找到预训练权重文件，使用随机初始化: {pretrained_path}")
    
    return model


def get_default_config(variant: str) -> Dict[str, Any]:
    """获取指定变体的默认配置"""
    
    base_config = {
        'input_size': 32,
        'dropout': 0.0
    }
    
    if variant == 'lenet5_mnist':
        config = {
            **base_config,
            'num_classes': 10,
            'input_channels': 1
        }
    elif variant == 'lenet5_cifar10':
        config = {
            **base_config,
            'num_classes': 10,
            'input_channels': 3
        }
    elif variant == 'lenet5_cifar100':
        config = {
            **base_config,
            'num_classes': 100,
            'input_channels': 3
        }
    else:
        raise ValueError(f"不支持的变体: {variant}")
    
    return config


def create_lenet_from_checkpoint(checkpoint_path: str) -> LeNet5:
    """
    从检查点文件创建LeNet模型
    
    Args:
        checkpoint_path: 检查点文件路径
    
    Returns:
        LeNet5模型实例
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    print(f"从检查点加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取配置
    if 'config' in checkpoint:
        model_config = checkpoint['config'].get('model', {})
    else:
        # 如果没有配置，尝试从模型权重推断
        model_config = infer_config_from_state_dict(checkpoint.get('model_state_dict', checkpoint))
    
    # 创建模型
    model = LeNet5(model_config)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"模型加载成功，参数量: {model.count_parameters():,}")
    
    return model


def infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """从状态字典推断模型配置"""
    
    # 推断输入通道数
    if 'conv1.weight' in state_dict:
        input_channels = state_dict['conv1.weight'].shape[1]
    else:
        input_channels = 1  # 默认值
    
    # 推断类别数
    if 'fc3.weight' in state_dict:
        num_classes = state_dict['fc3.weight'].shape[0]
    elif 'classifier.weight' in state_dict:
        num_classes = state_dict['classifier.weight'].shape[0]
    else:
        num_classes = 10  # 默认值
    
    config = {
        'num_classes': num_classes,
        'input_channels': input_channels,
        'input_size': 32,  # LeNet标准输入尺寸
        'dropout': 0.0
    }
    
    return config


def save_pretrained_weights(model: LeNet5, save_path: str, 
                          config: Dict[str, Any], 
                          training_info: Optional[Dict[str, Any]] = None):
    """
    保存预训练权重
    
    Args:
        model: 要保存的模型
        save_path: 保存路径
        config: 模型配置
        training_info: 训练信息（可选）
    """
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 准备保存内容
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': {'model': config},
        'model_info': {
            'architecture': 'LeNet-5',
            'parameters': model.count_parameters(),
            'input_size': config.get('input_size', 32),
            'num_classes': config.get('num_classes', 10),
            'input_channels': config.get('input_channels', 1)
        }
    }
    
    # 添加训练信息
    if training_info:
        save_dict.update(training_info)
    
    # 保存
    torch.save(save_dict, save_path)
    print(f"模型权重已保存到: {save_path}")


# 预定义的模型变体
LENET_VARIANTS = {
    'lenet5_mnist': {
        'description': 'LeNet-5 trained on MNIST dataset',
        'num_classes': 10,
        'input_channels': 1,
        'input_size': 32
    },
    'lenet5_cifar10': {
        'description': 'LeNet-5 trained on CIFAR-10 dataset',
        'num_classes': 10,
        'input_channels': 3,
        'input_size': 32
    },
    'lenet5_cifar100': {
        'description': 'LeNet-5 trained on CIFAR-100 dataset',
        'num_classes': 100,
        'input_channels': 3,
        'input_size': 32
    }
}


def list_available_models():
    """列出可用的预训练模型"""
    print("可用的LeNet预训练模型变体:")
    print("-" * 50)
    for variant, info in LENET_VARIANTS.items():
        print(f"{variant}:")
        print(f"  描述: {info['description']}")
        print(f"  类别数: {info['num_classes']}")
        print(f"  输入通道: {info['input_channels']}")
        print(f"  输入尺寸: {info['input_size']}x{info['input_size']}")
        print()


if __name__ == "__main__":
    # 测试加载功能
    print("测试LeNet预训练模型加载功能")
    
    # 列出可用模型
    list_available_models()
    
    # 测试创建模型
    model = load_pretrained_lenet('lenet5_mnist')
    print(f"创建LeNet-5模型成功，参数量: {model.count_parameters():,}")
    
    # 测试前向传播
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        output = model(x)
    print(f"测试前向传播成功，输出形状: {output.shape}")