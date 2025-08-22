"""
YOLOv3模型实现

YOLOv3是2018年提出的目标检测模型，在YOLOv2的基础上进行了重要改进，
包括使用Darknet-53骨干网络、多尺度预测和更好的损失函数。

论文: "YOLOv3: An Incremental Improvement" (2018)
作者: Joseph Redmon, Ali Farhadi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Dict, Any, List, Tuple, Optional

# 添加路径以导入基础模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
from shared.base_model import BaseModel


class ConvBlock(nn.Module):
    """YOLOv3基础卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """YOLOv3残差块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x


class Darknet53(nn.Module):
    """Darknet-53骨干网络"""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        # 初始卷积层
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64, stride=2)
        
        # 残差块
        self.res_block1 = self._make_layer(64, 1)
        self.conv3 = ConvBlock(64, 128, stride=2)
        self.res_block2 = self._make_layer(128, 2)
        self.conv4 = ConvBlock(128, 256, stride=2)
        self.res_block3 = self._make_layer(256, 8)
        self.conv5 = ConvBlock(256, 512, stride=2)
        self.res_block4 = self._make_layer(512, 8)
        self.conv6 = ConvBlock(512, 1024, stride=2)
        self.res_block5 = self._make_layer(1024, 4)
        
        # 分类层（如果需要）
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_layer(self, channels: int, num_blocks: int) -> nn.Sequential:
        """创建残差层"""
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, extract_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            extract_features: 是否提取多尺度特征用于检测
        
        Returns:
            如果extract_features=True，返回多尺度特征图
            否则返回分类结果
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res_block1(x)
        x = self.conv3(x)
        x = self.res_block2(x)
        x = self.conv4(x)
        x = self.res_block3(x)
        
        # 保存用于检测的特征图
        feat1 = x  # 256通道，用于小目标检测
        
        x = self.conv5(x)
        x = self.res_block4(x)
        
        feat2 = x  # 512通道，用于中等目标检测
        
        x = self.conv6(x)
        x = self.res_block5(x)
        
        feat3 = x  # 1024通道，用于大目标检测
        
        if extract_features:
            return {
                'feat1': feat1,  # [B, 256, H/8, W/8]
                'feat2': feat2,  # [B, 512, H/16, W/16]
                'feat3': feat3   # [B, 1024, H/32, W/32]
            }
        else:
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x


class YOLOv3Head(nn.Module):
    """YOLOv3检测头"""
    
    def __init__(self, in_channels: int, num_classes: int, anchors: List[Tuple[int, int]]):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = anchors
        
        # 每个anchor box预测: x, y, w, h, confidence, class_probs
        out_channels = self.num_anchors * (5 + num_classes)
        
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels, in_channels * 2),
            ConvBlock(in_channels * 2, in_channels),
            ConvBlock(in_channels, in_channels * 2),
            ConvBlock(in_channels * 2, in_channels),
            ConvBlock(in_channels, in_channels * 2),
        )
        
        self.conv_out = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.conv_out(x)
        return x


class YOLOv3(BaseModel):
    """
    YOLOv3目标检测模型
    
    Args:
        config: 模型配置字典
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_classes = config.get('num_classes', 80)
        self.img_size = config.get('img_size', 416)
        
        # YOLOv3使用的anchor boxes (3个尺度，每个尺度3个anchor)
        self.anchors = config.get('anchors', [
            # 大目标 (32x下采样)
            [(116, 90), (156, 198), (373, 326)],
            # 中等目标 (16x下采样)
            [(30, 61), (62, 45), (59, 119)],
            # 小目标 (8x下采样)
            [(10, 13), (16, 30), (33, 23)]
        ])
        
        # 骨干网络
        self.backbone = Darknet53()
        
        # 检测头
        self.head_large = YOLOv3Head(1024, self.num_classes, self.anchors[0])  # 大目标
        self.head_medium = YOLOv3Head(512, self.num_classes, self.anchors[1])  # 中等目标
        self.head_small = YOLOv3Head(256, self.num_classes, self.anchors[2])   # 小目标
        
        # 上采样和融合层
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_medium = ConvBlock(1024, 512, kernel_size=1, padding=0)
        self.conv_medium_merge = ConvBlock(1024, 512)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_small = ConvBlock(512, 256, kernel_size=1, padding=0)
        self.conv_small_merge = ConvBlock(512, 256)
        
        # 损失函数权重
        self.lambda_coord = config.get('lambda_coord', 5.0)
        self.lambda_noobj = config.get('lambda_noobj', 0.5)
        self.lambda_class = config.get('lambda_class', 1.0)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            三个尺度的预测结果列表
        """
        # 提取特征
        features = self.backbone(x, extract_features=True)
        feat1 = features['feat1']  # [B, 256, H/8, W/8]
        feat2 = features['feat2']  # [B, 512, H/16, W/16]
        feat3 = features['feat3']  # [B, 1024, H/32, W/32]
        
        # 大目标检测 (32x下采样)
        pred_large = self.head_large(feat3)  # [B, anchors*(5+classes), H/32, W/32]
        
        # 中等目标检测 (16x下采样)
        x_medium = self.conv_medium(feat3)
        x_medium = self.upsample1(x_medium)
        x_medium = torch.cat([x_medium, feat2], dim=1)
        x_medium = self.conv_medium_merge(x_medium)
        pred_medium = self.head_medium(x_medium)  # [B, anchors*(5+classes), H/16, W/16]
        
        # 小目标检测 (8x下采样)
        x_small = self.conv_small(x_medium)
        x_small = self.upsample2(x_small)
        x_small = torch.cat([x_small, feat1], dim=1)
        x_small = self.conv_small_merge(x_small)
        pred_small = self.head_small(x_small)  # [B, anchors*(5+classes), H/8, W/8]
        
        return [pred_large, pred_medium, pred_small]
    
    def decode_predictions(self, predictions: List[torch.Tensor], 
                          conf_threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        """
        解码预测结果
        
        Args:
            predictions: 模型预测结果
            conf_threshold: 置信度阈值
        
        Returns:
            解码后的边界框、置信度和类别
        """
        batch_size = predictions[0].size(0)
        results = []
        
        for b in range(batch_size):
            boxes = []
            scores = []
            classes = []
            
            for scale_idx, pred in enumerate(predictions):
                # 预测形状: [anchors*(5+classes), H, W]
                pred_b = pred[b]
                grid_h, grid_w = pred_b.shape[1], pred_b.shape[2]
                
                # 重新整理为 [H, W, anchors, 5+classes]
                pred_b = pred_b.view(len(self.anchors[scale_idx]), 5 + self.num_classes, grid_h, grid_w)
                pred_b = pred_b.permute(2, 3, 0, 1)  # [H, W, anchors, 5+classes]
                
                # 获取anchor信息
                anchors = self.anchors[scale_idx]
                stride = self.img_size // grid_h  # 下采样倍数
                
                # 创建网格
                grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                grid_x = grid_x.to(pred.device).float()
                grid_y = grid_y.to(pred.device).float()
                
                # 解码每个anchor
                for anchor_idx, (anchor_w, anchor_h) in enumerate(anchors):
                    # 提取预测
                    pred_anchor = pred_b[:, :, anchor_idx, :]  # [H, W, 5+classes]
                    
                    # 边界框坐标
                    x = (torch.sigmoid(pred_anchor[:, :, 0]) + grid_x) * stride
                    y = (torch.sigmoid(pred_anchor[:, :, 1]) + grid_y) * stride
                    w = torch.exp(pred_anchor[:, :, 2]) * anchor_w
                    h = torch.exp(pred_anchor[:, :, 3]) * anchor_h
                    
                    # 置信度
                    conf = torch.sigmoid(pred_anchor[:, :, 4])
                    
                    # 类别概率
                    class_probs = torch.sigmoid(pred_anchor[:, :, 5:])
                    
                    # 过滤低置信度
                    mask = conf > conf_threshold
                    if mask.sum() == 0:
                        continue
                    
                    # 提取有效预测
                    valid_x = x[mask]
                    valid_y = y[mask]
                    valid_w = w[mask]
                    valid_h = h[mask]
                    valid_conf = conf[mask]
                    valid_class_probs = class_probs[mask]
                    
                    # 转换为x1,y1,x2,y2格式
                    x1 = valid_x - valid_w / 2
                    y1 = valid_y - valid_h / 2
                    x2 = valid_x + valid_w / 2
                    y2 = valid_y + valid_h / 2
                    
                    # 组合边界框
                    valid_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                    
                    # 计算最终分数
                    class_scores, class_indices = torch.max(valid_class_probs, dim=1)
                    final_scores = valid_conf * class_scores
                    
                    boxes.append(valid_boxes)
                    scores.append(final_scores)
                    classes.append(class_indices)
            
            # 合并所有尺度的结果
            if boxes:
                all_boxes = torch.cat(boxes, dim=0)
                all_scores = torch.cat(scores, dim=0)
                all_classes = torch.cat(classes, dim=0)
            else:
                all_boxes = torch.empty((0, 4), device=predictions[0].device)
                all_scores = torch.empty((0,), device=predictions[0].device)
                all_classes = torch.empty((0,), device=predictions[0].device, dtype=torch.long)
            
            results.append({
                'boxes': all_boxes,
                'scores': all_scores,
                'classes': all_classes
            })
        
        return results


def create_yolov3(config: Optional[Dict[str, Any]] = None) -> YOLOv3:
    """
    创建YOLOv3模型
    
    Args:
        config: 模型配置
    
    Returns:
        YOLOv3模型实例
    """
    if config is None:
        config = {
            'num_classes': 80,  # COCO数据集
            'img_size': 416,
            'lambda_coord': 5.0,
            'lambda_noobj': 0.5,
            'lambda_class': 1.0
        }
    
    return YOLOv3(config)


if __name__ == "__main__":
    # 测试YOLOv3模型
    print("测试YOLOv3模型:")
    
    # 创建模型
    model = create_yolov3()
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 测试前向传播
    batch_size = 2
    img_size = 416
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    with torch.no_grad():
        predictions = model(x)
    
    print(f"输入形状: {x.shape}")
    for i, pred in enumerate(predictions):
        print(f"预测{i+1}形状: {pred.shape}")
    
    # 测试解码
    print("\\n测试预测解码:")
    results = model.decode_predictions(predictions, conf_threshold=0.1)
    
    for i, result in enumerate(results):
        print(f"样本{i+1}:")
        print(f"  检测框数量: {result['boxes'].size(0)}")
        print(f"  边界框形状: {result['boxes'].shape}")
        print(f"  分数形状: {result['scores'].shape}")
        print(f"  类别形状: {result['classes'].shape}")
    
    # 测试骨干网络
    print("\\n测试Darknet-53骨干网络:")
    backbone = Darknet53()
    backbone_params = sum(p.numel() for p in backbone.parameters())
    print(f"Darknet-53参数量: {backbone_params:,}")
    
    # 特征提取测试
    features = backbone(x, extract_features=True)
    for name, feat in features.items():
        print(f"{name}形状: {feat.shape}")