"""
计算机视觉模块

包含图像分类、目标检测等计算机视觉任务的模型实现。
"""

# 图像分类模型
from .image_classification.transformer.mae.model import MAE
from .image_classification.transformer.vit.model import VisionTransformer, DeiT
from .image_classification.cnn.lenet.model import LeNet5, LeNetVariant
from .image_classification.cnn.resnet.model import ResNet, PreActResNet

# 目标检测模型
from .object_detection.yolo_series.yolov1.model import YOLOv1

__all__ = [
    'MAE',
    'VisionTransformer',
    'DeiT', 
    'LeNet5',
    'LeNetVariant',
    'ResNet',
    'PreActResNet',
    'YOLOv1'
]