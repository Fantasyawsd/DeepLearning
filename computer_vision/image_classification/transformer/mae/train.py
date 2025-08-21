"""
MAE模型训练脚本

支持分布式训练、混合精度、学习率调度等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import time
import json
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# 添加路径以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

from computer_vision.image_classification.transformer.mae.model import MAE
from computer_vision.image_classification.transformer.mae.dataset import create_mae_dataloaders
from utils.logger import setup_logger
from utils.config import Config


class MAETrainer:
    """
    MAE模型训练器
    
    支持完整的训练流程，包括模型训练、验证、保存和恢复
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            device: 训练设备
        """
        self.config = config
        self.device = device
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        # 设置日志
        self.logger = setup_logger('MAE_Training', 
                                 log_file=os.path.join(config.get('output_dir', './outputs'), 'training.log'))
        
        # 创建输出目录
        self.output_dir = config.get('output_dir', './outputs/mae')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化优化器和调度器
        self._init_optimizer()
        
        # 初始化混合精度训练
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = GradScaler()
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def _init_model(self):
        """初始化模型"""
        self.model = MAE(self.config)
        self.model.to(self.device)
        
        # 分布式训练
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            self.model = DDP(self.model, device_ids=[self.device])
        
        self.logger.info(f"模型参数数量: {self.model.get_num_params():,}")
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        self.train_loader, self.val_loader = create_mae_dataloaders(self.config)
        
        self.logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        if self.val_loader:
            self.logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        # 优化器配置
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adamw')
        lr = optimizer_config.get('lr', 1.5e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.05)
        
        if optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.95))
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=optimizer_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        # 学习率调度器
        scheduler_config = self.config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'cosine')
        
        if scheduler_name.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=scheduler_config.get('min_lr', 0.0)
            )
        elif scheduler_name.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_name.lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, epoch: int) -> float:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 获取掩码比例
        mask_ratio = self.config.get('mask_ratio', 0.75)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    loss, pred, mask = self.model(images, mask_ratio)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, pred, mask = self.model(images, mask_ratio)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 记录中间结果
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
                )
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate_epoch(self, epoch: int) -> float:
        """
        验证一个epoch
        
        Args:
            epoch: 当前epoch
            
        Returns:
            平均验证损失
        """
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        mask_ratio = self.config.get('mask_ratio', 0.75)
        
        with torch.no_grad():
            for images, _ in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        loss, pred, mask = self.model(images, mask_ratio)
                else:
                    loss, pred, mask = self.model(images, mask_ratio)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.output_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'保存最佳模型到 {best_path}')
        
        # 定期保存epoch检查点
        if epoch % self.config.get('save_interval', 50) == 0:
            epoch_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f'检查点文件不存在: {checkpoint_path}')
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        self.logger.info(f'从epoch {self.start_epoch}恢复训练，最佳损失: {self.best_loss:.4f}')
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='训练损失')
        if self.val_losses and len(self.val_losses) > 0:
            axes[0, 0].plot(epochs, self.val_losses, 'r-', label='验证损失')
        axes[0, 0].set_title('训练/验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率
        axes[0, 1].plot(epochs, self.learning_rates, 'g-')
        axes[0, 1].set_title('学习率变化')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # 损失对数图
        axes[1, 0].semilogy(epochs, self.train_losses, 'b-', label='训练损失')
        if self.val_losses and len(self.val_losses) > 0:
            axes[1, 0].semilogy(epochs, self.val_losses, 'r-', label='验证损失')
        axes[1, 0].set_title('训练/验证损失 (对数)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率对数图
        axes[1, 1].semilogy(epochs, self.learning_rates, 'g-')
        axes[1, 1].set_title('学习率变化 (对数)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate (log)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    def train(self):
        """开始训练"""
        epochs = self.config.get('epochs', 100)
        
        self.logger.info(f'开始训练MAE模型，总共{epochs}个epoch')
        self.logger.info(f'设备: {self.device}')
        self.logger.info(f'输出目录: {self.output_dir}')
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate_epoch(epoch)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 检查是否为最佳模型
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            epoch_time = time.time() - epoch_start
            
            # 记录epoch结果
            self.logger.info(
                f'Epoch {epoch}/{epochs-1} - '
                f'训练损失: {train_loss:.4f}, '
                f'验证损失: {val_loss:.4f}, '
                f'最佳损失: {self.best_loss:.4f}, '
                f'时间: {epoch_time:.2f}s'
            )
            
            # 绘制训练曲线
            if epoch % self.config.get('plot_interval', 10) == 0:
                self.plot_training_curves()
        
        total_time = time.time() - start_time
        self.logger.info(f'训练完成！总用时: {total_time/3600:.2f}小时')
        
        # 保存最终训练曲线
        self.plot_training_curves()
        
        # 保存训练配置和结果
        results = {
            'config': self.config,
            'best_loss': self.best_loss,
            'total_epochs': epochs,
            'total_time': total_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        results_path = os.path.join(self.output_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def train_mae(config_path: str, resume: Optional[str] = None):
    """
    训练MAE模型的主函数
    
    Args:
        config_path: 配置文件路径
        resume: 恢复训练的检查点路径
    """
    # 加载配置
    config = Config.from_file(config_path)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建训练器
    trainer = MAETrainer(config.to_dict(), device)
    
    # 恢复训练
    if resume:
        trainer.load_checkpoint(resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练MAE模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 更新输出目录
    if args.output_dir:
        config = Config.from_file(args.config)
        config.output_dir = args.output_dir
        config.save(args.config)
    
    train_mae(args.config, args.resume)