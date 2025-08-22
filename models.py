"""
深度学习模型包

本包提供了按领域组织的深度学习模型实现。

项目结构：
- computer_vision/: 计算机视觉模型
  - image_classification/: 图像分类
    - cnn/: CNN系列模型 (LeNet, ResNet等)
    - transformer/: Transformer系列模型 (ViT, MAE等)
  - object_detection/: 目标检测模型 (YOLO等)
- nlp/: 自然语言处理模型
  - language_models/: 语言模型 (GPT等)
"""

# 导入主要模型以保持向后兼容性
from computer_vision import MAE, VisionTransformer, LeNet5, ResNet, YOLOv1
from nlp import GPT

# 为常用模型提供别名
ViT = VisionTransformer
LeNet = LeNet5

__all__ = [
    # 计算机视觉
    'MAE',
    'VisionTransformer',
    'ViT',  # 别名
    'LeNet5',
    'LeNet',  # 别名
    'ResNet',
    'YOLOv1',
    
    # 自然语言处理
    'GPT',
]

# 版本信息
__version__ = '1.0.0'