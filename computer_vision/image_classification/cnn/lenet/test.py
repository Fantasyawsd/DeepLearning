"""
LeNet测试和推理脚本

提供模型测试和推理功能，包括：
- 测试集评估
- 单张图像推理
- 模型性能分析
- 可视化结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import sys
from tqdm import tqdm
import time

# 添加路径以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

from model import LeNet5
from dataset import create_dataloader
from utils.metrics import AverageMeter, accuracy


class LeNetTester:
    """LeNet测试器"""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 self.config.get('device', {}).get('use_cuda', True) else 'cpu')
        
        # 创建模型
        self.model = LeNet5(self.config['model']).to(self.device)
        
        # 加载检查点
        self.load_checkpoint(checkpoint_path)
        
        # 创建数据加载器
        _, _, self.test_loader = create_dataloader(self.config)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 类别名称
        from dataset import LeNetDataset
        dataset_handler = LeNetDataset(self.config)
        self.class_names = dataset_handler.get_class_names()
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if os.path.isfile(checkpoint_path):
            print(f'加载检查点: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f'检查点加载成功 (epoch {checkpoint.get("epoch", "unknown")}, '
                  f'best_acc {checkpoint.get("best_acc", 0.0):.2f}%)')
        else:
            raise FileNotFoundError(f'未找到检查点文件: {checkpoint_path}')
    
    def test(self):
        """测试模型性能"""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        # 用于计算混淆矩阵
        all_predictions = []
        all_targets = []
        
        print("开始测试...")
        start_time = time.time()
        
        with torch.no_grad():
            with tqdm(self.test_loader, desc='测试') as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 前向传播
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # 计算精度
                    acc1 = accuracy(output, target, topk=(1,))[0]
                    
                    # 更新统计
                    losses.update(loss.item(), data.size(0))
                    top1.update(acc1.item(), data.size(0))
                    
                    # 保存预测结果
                    predictions = torch.argmax(output, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'Loss': f'{losses.avg:.4f}',
                        'Acc': f'{top1.avg:.2f}%'
                    })
        
        test_time = time.time() - start_time
        
        print(f"\\n测试完成!")
        print(f"测试损失: {losses.avg:.4f}")
        print(f"测试精度: {top1.avg:.2f}%")
        print(f"测试时间: {test_time:.2f}秒")
        print(f"平均推理时间: {test_time / len(self.test_loader.dataset) * 1000:.2f}ms/样本")
        
        return losses.avg, top1.avg, all_predictions, all_targets
    
    def predict_single(self, image: torch.Tensor) -> dict:
        """单张图像推理"""
        self.model.eval()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)  # 添加batch维度
        
        image = image.to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            output = self.model(image)
            inference_time = time.time() - start_time
            
            # 计算概率
            probabilities = F.softmax(output, dim=1)
            
            # 获取预测结果
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # 获取top-5预测
            top5_probs, top5_indices = torch.topk(probabilities[0], 
                                                 min(5, len(self.class_names)))
            
            top5_results = []
            for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
                class_name = self.class_names[idx.item()] if idx.item() < len(self.class_names) else f"类别{idx.item()}"
                top5_results.append({
                    'rank': i + 1,
                    'class_id': idx.item(),
                    'class_name': class_name,
                    'probability': prob.item()
                })
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"类别{predicted_class}",
            'confidence': confidence,
            'inference_time': inference_time,
            'top5_predictions': top5_results
        }
    
    def analyze_performance(self, predictions: list, targets: list):
        """分析模型性能"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算每个类别的精度
        num_classes = len(self.class_names)
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        
        for i in range(len(targets)):
            class_total[targets[i]] += 1
            if predictions[i] == targets[i]:
                class_correct[targets[i]] += 1
        
        print("\\n各类别精度分析:")
        print("-" * 50)
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                class_name = self.class_names[i] if i < len(self.class_names) else f"类别{i}"
                print(f"{class_name}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    def visualize_predictions(self, num_samples: int = 8):
        """可视化预测结果"""
        self.model.eval()
        
        # 获取一些测试样本
        data_iter = iter(self.test_loader)
        images, labels = next(data_iter)
        
        # 选择前num_samples个样本
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            predictions = torch.argmax(outputs, dim=1).cpu()
            probabilities = F.softmax(outputs, dim=1).cpu()
        
        # 创建可视化
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(num_samples):
            # 获取图像
            img = images[i]
            if img.shape[0] == 1:  # 灰度图
                img = img.squeeze(0)
                cmap = 'gray'
            else:  # RGB图
                img = img.permute(1, 2, 0)
                cmap = None
            
            # 反归一化显示
            if self.config['data']['dataset'] == 'MNIST':
                img = img * 0.3081 + 0.1307
            
            # 显示图像
            axes[i].imshow(img, cmap=cmap)
            axes[i].axis('off')
            
            # 设置标题
            true_label = self.class_names[labels[i].item()] if labels[i].item() < len(self.class_names) else f"类别{labels[i].item()}"
            pred_label = self.class_names[predictions[i].item()] if predictions[i].item() < len(self.class_names) else f"类别{predictions[i].item()}"
            confidence = probabilities[i, predictions[i].item()].item()
            
            color = 'green' if predictions[i] == labels[i] else 'red'
            axes[i].set_title(f'真实: {true_label}\\n预测: {pred_label}\\n置信度: {confidence:.2f}', 
                             color=color, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('lenet_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("预测结果已保存到 lenet_predictions.png")


def main():
    parser = argparse.ArgumentParser(description='LeNet测试脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--visualize', action='store_true', help='是否可视化预测结果')
    args = parser.parse_args()
    
    # 创建测试器
    tester = LeNetTester(args.config, args.checkpoint)
    
    # 测试模型
    test_loss, test_acc, predictions, targets = tester.test()
    
    # 性能分析
    tester.analyze_performance(predictions, targets)
    
    # 可视化
    if args.visualize:
        tester.visualize_predictions()


if __name__ == '__main__':
    main()