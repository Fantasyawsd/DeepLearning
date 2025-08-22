"""
LeNet训练脚本

提供完整的训练流程，包括：
- 模型训练和验证
- 检查点保存和恢复
- 学习率调度
- TensorBoard日志记录
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import os
import time
from tqdm import tqdm
import sys

# 添加路径以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

from model import LeNet5
from dataset import create_dataloader
from utils.logger import setup_logger
from utils.metrics import AverageMeter, accuracy


class LeNetTrainer:
    """LeNet训练器"""
    
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 self.config.get('device', {}).get('use_cuda', True) else 'cpu')
        
        # 创建模型
        self.model = LeNet5(self.config['model']).to(self.device)
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_dataloader(self.config)
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 创建保存目录
        self.checkpoint_dir = self.config.get('save', {}).get('checkpoint_dir', './checkpoints/lenet')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 设置日志
        log_dir = self.config.get('logging', {}).get('log_dir', './logs/lenet')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logger('LeNet', os.path.join(log_dir, 'train.log'))
        
        # TensorBoard
        if self.config.get('logging', {}).get('tensorboard', True):
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # 训练状态
        self.start_epoch = 0
        self.best_acc = 0.0
        
    def _create_optimizer(self):
        """创建优化器"""
        opt_config = self.config.get('training', {}).get('optimizer', {})
        opt_type = opt_config.get('type', 'Adam')
        lr = self.config.get('training', {}).get('learning_rate', 0.001)
        weight_decay = self.config.get('training', {}).get('weight_decay', 1e-4)
        
        if opt_type == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == 'SGD':
            momentum = opt_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {opt_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        sched_config = self.config.get('training', {}).get('scheduler', {})
        sched_type = sched_config.get('type', 'StepLR')
        
        if sched_type == 'StepLR':
            step_size = sched_config.get('step_size', 15)
            gamma = sched_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sched_type == 'CosineAnnealingLR':
            T_max = self.config.get('training', {}).get('epochs', 50)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif sched_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5)
        else:
            return None
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        print_freq = self.config.get('logging', {}).get('print_frequency', 100)
        
        with tqdm(self.train_loader, desc=f'训练 Epoch {epoch}') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 计算精度
                acc1 = accuracy(output, target, topk=(1,))[0]
                
                # 更新统计
                losses.update(loss.item(), data.size(0))
                top1.update(acc1.item(), data.size(0))
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc': f'{top1.avg:.2f}%'
                })
                
                # 记录日志
                if batch_idx % print_freq == 0:
                    self.logger.info(f'训练 Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                                   f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                                   f'Loss: {loss.item():.6f}\tAcc: {acc1.item():.2f}%')
                
                # TensorBoard记录
                if self.writer:
                    step = epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('Train/Loss', loss.item(), step)
                    self.writer.add_scalar('Train/Accuracy', acc1.item(), step)
                    self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], step)
        
        return losses.avg, top1.avg
    
    def validate(self, epoch: int):
        """验证模型"""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f'验证 Epoch {epoch}') as pbar:
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
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'Loss': f'{losses.avg:.4f}',
                        'Acc': f'{top1.avg:.2f}%'
                    })
        
        self.logger.info(f'验证 Epoch {epoch}: Loss: {losses.avg:.4f}, Acc: {top1.avg:.2f}%')
        
        # TensorBoard记录
        if self.writer:
            self.writer.add_scalar('Val/Loss', losses.avg, epoch)
            self.writer.add_scalar('Val/Accuracy', top1.avg, epoch)
        
        return losses.avg, top1.avg
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_acc': self.best_acc,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(state, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(state, best_path)
            self.logger.info(f'保存最佳模型到 {best_path}')
        
        # 定期保存
        save_freq = self.config.get('save', {}).get('save_frequency', 10)
        if epoch % save_freq == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(state, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if os.path.isfile(checkpoint_path):
            self.logger.info(f'加载检查点 {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_acc = checkpoint.get('best_acc', 0.0)
            
            self.logger.info(f'检查点加载成功 (epoch {self.start_epoch}, best_acc {self.best_acc:.2f}%)')
        else:
            self.logger.info(f'未找到检查点 {checkpoint_path}')
    
    def train(self):
        """完整训练流程"""
        num_epochs = self.config.get('training', {}).get('epochs', 50)
        eval_freq = self.config.get('validation', {}).get('eval_frequency', 1)
        save_best_only = self.config.get('save', {}).get('save_best_only', True)
        
        self.logger.info(f'开始训练，总共 {num_epochs} epochs')
        self.logger.info(f'设备: {self.device}')
        self.logger.info(f'模型参数量: {self.model.count_parameters():,}')
        
        for epoch in range(self.start_epoch, num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = 0.0, 0.0
            if epoch % eval_freq == 0:
                val_loss, val_acc = self.validate(epoch)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc if val_acc > 0 else train_acc)
                else:
                    self.scheduler.step()
            
            # 检查是否是最佳模型
            current_acc = val_acc if val_acc > 0 else train_acc
            is_best = current_acc > self.best_acc
            if is_best:
                self.best_acc = current_acc
            
            # 保存检查点
            if not save_best_only or is_best:
                self.save_checkpoint(epoch, is_best)
            
            self.logger.info(f'Epoch {epoch}: 训练Loss: {train_loss:.4f}, 训练Acc: {train_acc:.2f}%, '
                           f'验证Loss: {val_loss:.4f}, 验证Acc: {val_acc:.2f}%, '
                           f'最佳Acc: {self.best_acc:.2f}%')
        
        self.logger.info(f'训练完成！最佳验证精度: {self.best_acc:.2f}%')
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='LeNet训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 创建训练器
    trainer = LeNetTrainer(args.config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()