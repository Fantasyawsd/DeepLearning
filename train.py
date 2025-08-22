"""
深度学习模型训练脚本

支持新的域分类结构下的所有模型训练。
请使用各个模型目录下的专用训练脚本获得更好的训练体验。
"""

import argparse
import yaml
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MAE, ViT, LeNet, ResNet, GPT
from utils import Config, get_logger


def get_model(config: Config):
    """根据配置获取模型"""
    model_name = config.get('model_name', '').lower()
    
    if model_name == 'mae':
        return MAE(config.to_dict())
    elif model_name in ['vit', 'vision_transformer']:
        return ViT(config.to_dict())
    elif model_name == 'lenet':
        return LeNet(config.to_dict())
    elif model_name == 'resnet':
        return ResNet(config.to_dict())
    elif model_name == 'gpt':
        return GPT(config.to_dict())
    else:
        raise ValueError(f"未知模型: {model_name}. 支持的模型: mae, vit, lenet, resnet, gpt")


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='深度学习模型训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model', type=str, help='模型名称 (覆盖配置文件中的设置)')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if args.model:
        config_dict['model_name'] = args.model
    
    config = Config(config_dict)
    
    # 获取模型
    try:
        model = get_model(config)
        print(f"成功加载模型: {config.get('model_name')}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 推荐使用专用训练脚本
        model_name = config.get('model_name', '').lower()
        if model_name == 'mae':
            print("建议使用: computer_vision/image_classification/transformer/mae/train.py")
        elif model_name in ['vit', 'vision_transformer']:
            print("建议使用: computer_vision/image_classification/transformer/vit/train.py")
        elif model_name == 'gpt':
            print("建议使用: nlp/language_models/gpt_series/gpt/train.py")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())