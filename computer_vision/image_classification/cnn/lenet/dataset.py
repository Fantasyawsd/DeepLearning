"""
LeNet数据集处理模块

支持MNIST、CIFAR-10、CIFAR-100等数据集的加载和预处理
适配LeNet-5模型的输入要求
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any
import os


class LeNetDataset:
    """LeNet数据集处理类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config.get('data', {})
        self.dataset_name = self.data_config.get('dataset', 'MNIST')
        self.data_root = self.data_config.get('data_root', './data')
        self.batch_size = config.get('training', {}).get('batch_size', 128)
        self.num_workers = self.data_config.get('num_workers', 4)
        self.pin_memory = self.data_config.get('pin_memory', True)
        
        # 创建数据目录
        os.makedirs(self.data_root, exist_ok=True)
        
    def get_transforms(self, is_train: bool = True) -> transforms.Compose:
        """获取数据变换"""
        aug_config = self.data_config.get('augmentation', {})
        norm_config = aug_config.get('normalize', {})
        
        transform_list = []
        
        if self.dataset_name == 'MNIST':
            # MNIST: 28x28 -> 32x32 (LeNet原始输入尺寸)
            transform_list.extend([
                transforms.Resize(32),
                transforms.ToTensor()
            ])
            
            # 数据增强 (仅训练时)
            if is_train and aug_config.get('enabled', True):
                if aug_config.get('rotation', 0) > 0:
                    transform_list.insert(-1, transforms.RandomRotation(aug_config['rotation']))
            
            # 归一化
            mean = norm_config.get('mean', [0.1307])
            std = norm_config.get('std', [0.3081])
            transform_list.append(transforms.Normalize(mean, std))
            
        elif self.dataset_name in ['CIFAR10', 'CIFAR100']:
            # CIFAR: 32x32 (已是LeNet输入尺寸)
            transform_list.append(transforms.ToTensor())
            
            # 数据增强 (仅训练时)
            if is_train and aug_config.get('enabled', True):
                if aug_config.get('horizontal_flip', False):
                    transform_list.insert(-1, transforms.RandomHorizontalFlip())
                if aug_config.get('rotation', 0) > 0:
                    transform_list.insert(-1, transforms.RandomRotation(aug_config['rotation']))
            
            # 归一化
            if self.dataset_name == 'CIFAR10':
                mean = norm_config.get('mean', [0.4914, 0.4822, 0.4465])
                std = norm_config.get('std', [0.2023, 0.1994, 0.2010])
            else:  # CIFAR100
                mean = norm_config.get('mean', [0.5071, 0.4867, 0.4408])
                std = norm_config.get('std', [0.2675, 0.2565, 0.2761])
            
            transform_list.append(transforms.Normalize(mean, std))
        
        return transforms.Compose(transform_list)
    
    def get_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """获取训练和测试数据集"""
        train_transform = self.get_transforms(is_train=True)
        test_transform = self.get_transforms(is_train=False)
        
        if self.dataset_name == 'MNIST':
            train_dataset = torchvision.datasets.MNIST(
                root=self.data_root, train=True, download=True, transform=train_transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root=self.data_root, train=False, download=True, transform=test_transform
            )
        elif self.dataset_name == 'CIFAR10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_root, train=True, download=True, transform=train_transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_root, train=False, download=True, transform=test_transform
            )
        elif self.dataset_name == 'CIFAR100':
            train_dataset = torchvision.datasets.CIFAR100(
                root=self.data_root, train=True, download=True, transform=train_transform
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.data_root, train=False, download=True, transform=test_transform
            )
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        return train_dataset, test_dataset
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取数据加载器"""
        train_dataset, test_dataset = self.get_dataset()
        
        # 划分验证集
        val_split = self.config.get('validation', {}).get('val_split', 0.1)
        if val_split > 0:
            val_size = int(len(train_dataset) * val_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        else:
            val_dataset = None
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_names(self) -> list:
        """获取类别名称"""
        if self.dataset_name == 'MNIST':
            return [str(i) for i in range(10)]
        elif self.dataset_name == 'CIFAR10':
            return ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
        elif self.dataset_name == 'CIFAR100':
            return [f'类别{i}' for i in range(100)]  # CIFAR100类别太多，简化显示
        else:
            return []


def create_dataloader(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """便捷函数：创建数据加载器"""
    dataset_handler = LeNetDataset(config)
    return dataset_handler.get_dataloaders()


if __name__ == "__main__":
    # 测试数据加载
    import yaml
    
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader, test_loader = create_dataloader(config)
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset) if val_loader else 0}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 查看数据样本
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"批次 {batch_idx}: 数据形状 {data.shape}, 标签形状 {target.shape}")
        break