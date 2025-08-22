"""
YOLOv1 (You Only Look Once) 模型实现

YOLO是第一个端到端的实时目标检测系统，将目标检测问题转化为回归问题。

论文: "You Only Look Once: Unified, Real-Time Object Detection" (CVPR 2016)
作者: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, Optional, Tuple, List

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from shared.base_model import BaseModel


class YOLOv1(BaseModel):
    """
    YOLOv1 网络实现
    
    网络结构基于GoogLeNet，但使用1x1和3x3卷积替代Inception模块
    
    Args:
        num_classes: 目标类别数
        grid_size: 网格大小 (S x S)
        num_boxes: 每个网格预测的边界框数量 (B)
        input_size: 输入图像尺寸
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置获取参数
        self.num_classes = config.get('num_classes', 20)  # PASCAL VOC默认20类
        self.grid_size = config.get('grid_size', 7)  # 7x7网格
        self.num_boxes = config.get('num_boxes', 2)  # 每个网格预测2个框
        self.input_size = config.get('input_size', 448)
        
        # 特征提取骨干网络
        self.features = self._make_features()
        
        # 分类和回归头
        # 输出: (grid_size^2) * (num_boxes * 5 + num_classes)
        # 每个框: (x, y, w, h, confidence)
        output_size = self.grid_size * self.grid_size * (self.num_boxes * 5 + self.num_classes)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_size),
            nn.Sigmoid()  # 输出0-1之间的值
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_features(self) -> nn.Module:
        """构建特征提取网络"""
        layers = []
        in_channels = 3
        
        # YOLO特征提取网络配置
        # 格式: (out_channels, kernel_size, stride, padding, 'M'表示MaxPool)
        config = [
            (64, 7, 2, 3), 'M',
            (192, 3, 1, 1), 'M',
            (128, 1, 1, 0), (256, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1), 'M',
            (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1),
            (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1),
            (512, 1, 1, 0), (1024, 3, 1, 1), 'M',
            (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0), (1024, 3, 1, 1),
            (1024, 3, 1, 1), (1024, 3, 2, 1),
            (1024, 3, 1, 1), (1024, 3, 1, 1)
        ]
        
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, kernel_size, stride, padding = v
                conv = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=kernel_size, stride=stride, padding=padding)
                layers.append(conv)
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.LeakyReLU(0.1, inplace=True))
                in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, 3, 448, 448)
            
        Returns:
            预测张量 (batch_size, grid_size, grid_size, num_boxes*5 + num_classes)
        """
        # 特征提取
        x = self.features(x)  # (batch_size, 1024, 7, 7)
        
        # 展平
        x = x.view(x.size(0), -1)  # (batch_size, 1024*7*7)
        
        # 分类和回归
        x = self.classifier(x)  # (batch_size, S*S*(B*5+C))
        
        # 重塑为网格形式
        batch_size = x.size(0)
        x = x.view(batch_size, self.grid_size, self.grid_size, 
                  self.num_boxes * 5 + self.num_classes)
        
        return x
    
    def decode_predictions(self, predictions: torch.Tensor, 
                         confidence_threshold: float = 0.5) -> List[Dict]:
        """
        解码预测结果
        
        Args:
            predictions: 模型输出 (batch_size, S, S, B*5+C)
            confidence_threshold: 置信度阈值
            
        Returns:
            解码后的边界框列表
        """
        batch_size = predictions.size(0)
        results = []
        
        for batch_idx in range(batch_size):
            pred = predictions[batch_idx]  # (S, S, B*5+C)
            batch_boxes = []
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell_pred = pred[i, j]  # (B*5+C,)
                    
                    # 提取类别概率
                    class_probs = cell_pred[self.num_boxes * 5:]  # (C,)
                    
                    # 提取边界框
                    for box_idx in range(self.num_boxes):
                        start_idx = box_idx * 5
                        box_pred = cell_pred[start_idx:start_idx + 5]  # (5,)
                        
                        x, y, w, h, confidence = box_pred
                        
                        if confidence > confidence_threshold:
                            # 转换坐标到图像坐标系
                            center_x = (j + x) / self.grid_size
                            center_y = (i + y) / self.grid_size
                            width = w
                            height = h
                            
                            # 计算每个类别的得分
                            for class_idx in range(self.num_classes):
                                class_score = confidence * class_probs[class_idx]
                                
                                if class_score > confidence_threshold:
                                    batch_boxes.append({
                                        'class_id': class_idx,
                                        'confidence': float(class_score),
                                        'bbox': [
                                            float(center_x - width/2),   # x1
                                            float(center_y - height/2),  # y1
                                            float(center_x + width/2),   # x2
                                            float(center_y + height/2)   # y2
                                        ]
                                    })
            
            results.append(batch_boxes)
        
        return results
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class YOLOLoss(nn.Module):
    """
    YOLO损失函数实现
    
    包含定位损失、置信度损失和分类损失
    """
    
    def __init__(self, grid_size: int = 7, num_boxes: int = 2, num_classes: int = 20,
                 lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord  # 坐标损失权重
        self.lambda_noobj = lambda_noobj  # 无目标置信度损失权重
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算YOLO损失
        
        Args:
            predictions: 模型预测 (batch_size, S, S, B*5+C)
            targets: 真实标签 (batch_size, S, S, B*5+C)
            
        Returns:
            总损失
        """
        batch_size = predictions.size(0)
        
        # 分离预测的不同部分
        # 边界框: (x, y, w, h, confidence) * num_boxes
        # 类别: num_classes
        
        # 计算有目标的网格
        # targets中最后一个维度的格式与predictions相同
        obj_mask = targets[..., 4] > 0  # (batch_size, S, S)
        obj_mask = obj_mask.unsqueeze(-1)  # (batch_size, S, S, 1)
        
        # 定位损失 (x, y, w, h)
        coord_loss = 0.0
        confidence_loss = 0.0
        
        for box_idx in range(self.num_boxes):
            start_idx = box_idx * 5
            
            # 预测的边界框参数
            pred_x = predictions[..., start_idx]
            pred_y = predictions[..., start_idx + 1]
            pred_w = predictions[..., start_idx + 2]
            pred_h = predictions[..., start_idx + 3]
            pred_conf = predictions[..., start_idx + 4]
            
            # 真实的边界框参数
            true_x = targets[..., start_idx]
            true_y = targets[..., start_idx + 1] 
            true_w = targets[..., start_idx + 2]
            true_h = targets[..., start_idx + 3]
            true_conf = targets[..., start_idx + 4]
            
            # 坐标损失 (只对有目标的网格计算)
            obj_mask_box = (true_conf > 0).float()
            
            coord_loss += torch.sum(obj_mask_box * (pred_x - true_x)**2)
            coord_loss += torch.sum(obj_mask_box * (pred_y - true_y)**2)
            coord_loss += torch.sum(obj_mask_box * (torch.sqrt(pred_w + 1e-8) - torch.sqrt(true_w + 1e-8))**2)
            coord_loss += torch.sum(obj_mask_box * (torch.sqrt(pred_h + 1e-8) - torch.sqrt(true_h + 1e-8))**2)
            
            # 置信度损失
            # 有目标的网格
            confidence_loss += torch.sum(obj_mask_box * (pred_conf - true_conf)**2)
            
            # 无目标的网格
            noobj_mask = (true_conf == 0).float()
            confidence_loss += self.lambda_noobj * torch.sum(noobj_mask * pred_conf**2)
        
        # 分类损失
        pred_classes = predictions[..., self.num_boxes * 5:]  # (batch_size, S, S, C)
        true_classes = targets[..., self.num_boxes * 5:]      # (batch_size, S, S, C)
        
        obj_mask_class = obj_mask.expand_as(pred_classes)
        class_loss = torch.sum(obj_mask_class * (pred_classes - true_classes)**2)
        
        # 总损失
        total_loss = (self.lambda_coord * coord_loss + 
                     confidence_loss + 
                     class_loss) / batch_size
        
        return total_loss


def create_yolo_v1(config: Optional[Dict[str, Any]] = None) -> YOLOv1:
    """
    创建YOLOv1模型的便捷函数
    
    Args:
        config: 模型配置
        
    Returns:
        YOLOv1模型实例
    """
    default_config = {
        'num_classes': 20,  # PASCAL VOC
        'grid_size': 7,
        'num_boxes': 2,
        'input_size': 448
    }
    
    if config:
        default_config.update(config)
    
    return YOLOv1(default_config)


# 数据集配置
YOLO_DATASET_CONFIGS = {
    'pascal_voc': {
        'num_classes': 20,
        'class_names': [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    },
    'coco': {
        'num_classes': 80,
        'class_names': None  # COCO有80个类别，这里省略
    }
}


def get_yolo_config(dataset: str) -> Dict[str, Any]:
    """获取特定数据集的YOLO配置"""
    base_config = {
        'grid_size': 7,
        'num_boxes': 2,
        'input_size': 448
    }
    
    dataset_config = YOLO_DATASET_CONFIGS.get(dataset, YOLO_DATASET_CONFIGS['pascal_voc'])
    base_config.update(dataset_config)
    
    return base_config


def non_maximum_suppression(boxes: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    非极大值抑制 (NMS)
    
    Args:
        boxes: 边界框列表，每个包含'bbox'和'confidence'
        iou_threshold: IoU阈值
        
    Returns:
        NMS后的边界框列表
    """
    if not boxes:
        return []
    
    # 按置信度排序
    boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while boxes:
        # 取置信度最高的框
        current = boxes.pop(0)
        keep.append(current)
        
        # 计算与其他框的IoU
        remaining = []
        for box in boxes:
            iou = calculate_iou(current['bbox'], box['bbox'])
            if iou < iou_threshold:
                remaining.append(box)
        
        boxes = remaining
    
    return keep


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        box1, box2: [x1, y1, x2, y2] 格式的边界框
        
    Returns:
        IoU值
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0