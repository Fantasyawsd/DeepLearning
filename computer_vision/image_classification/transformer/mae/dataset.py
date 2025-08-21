"""
MAE数据集处理和预处理模块

支持常见的图像数据集，包括ImageNet、CIFAR-10、CIFAR-100等
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from typing import Tuple, Optional, Callable, List
import numpy as np


class ImageNetDataset(Dataset):
    """
    ImageNet数据集加载器
    
    支持ImageNet-1K数据集的加载和预处理
    """
    
    def __init__(self, root: str, split: str = 'train', transform: Optional[Callable] = None):
        """
        Args:
            root: 数据集根目录
            split: 数据集分割 ('train', 'val')
            transform: 数据变换
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        self.data_dir = os.path.join(root, split)
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, target


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10数据集加载器
    """
    
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, download: bool = True):
        """
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 数据变换
            download: 是否下载数据集
        """
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class CIFAR100Dataset(Dataset):
    """
    CIFAR-100数据集加载器
    """
    
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, download: bool = True):
        """
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 数据变换
            download: 是否下载数据集
        """
        self.dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, transform=transform, download=download
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class CustomImageDataset(Dataset):
    """
    自定义图像数据集加载器
    
    支持自定义目录结构的图像数据集
    """
    
    def __init__(self, root: str, transform: Optional[Callable] = None, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')):
        """
        Args:
            root: 数据集根目录
            transform: 数据变换
            extensions: 支持的文件扩展名
        """
        self.root = root
        self.transform = transform
        self.extensions = extensions
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """加载数据样本"""
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(self.extensions):
                    self.samples.append(os.path.join(root, file))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


def get_mae_transforms(img_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    获取MAE模型的数据变换
    
    Args:
        img_size: 图像尺寸
        is_training: 是否为训练模式
        
    Returns:
        数据变换组合
    """
    if is_training:
        # 训练时的数据增强
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=3),  # 3是BICUBIC
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # 验证/测试时的变换
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 8/7), interpolation=3),  # to maintain same ratio w.r.t. 224 images
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    return transform


def get_augmentation_transforms(img_size: int = 224, augment_strength: str = 'medium') -> transforms.Compose:
    """
    获取数据增强变换
    
    Args:
        img_size: 图像尺寸
        augment_strength: 增强强度 ('light', 'medium', 'strong')
        
    Returns:
        数据增强变换组合
    """
    base_transforms = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
    ]
    
    if augment_strength == 'light':
        augment_transforms = [
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ]
    elif augment_strength == 'medium':
        augment_transforms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
        ]
    elif augment_strength == 'strong':
        augment_transforms = [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        ]
    else:
        augment_transforms = []
    
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    return transforms.Compose(base_transforms + augment_transforms + final_transforms)


def create_mae_dataloader(dataset_name: str, data_root: str, img_size: int = 224, 
                         batch_size: int = 32, num_workers: int = 4, 
                         is_training: bool = True) -> DataLoader:
    """
    创建MAE数据加载器
    
    Args:
        dataset_name: 数据集名称 ('imagenet', 'cifar10', 'cifar100', 'custom')
        data_root: 数据集根目录
        img_size: 图像尺寸
        batch_size: 批次大小
        num_workers: 工作进程数
        is_training: 是否为训练模式
        
    Returns:
        数据加载器
    """
    transform = get_mae_transforms(img_size, is_training)
    
    if dataset_name.lower() == 'imagenet':
        split = 'train' if is_training else 'val'
        dataset = ImageNetDataset(data_root, split=split, transform=transform)
    elif dataset_name.lower() == 'cifar10':
        dataset = CIFAR10Dataset(data_root, train=is_training, transform=transform)
    elif dataset_name.lower() == 'cifar100':
        dataset = CIFAR100Dataset(data_root, train=is_training, transform=transform)
    elif dataset_name.lower() == 'custom':
        dataset = CustomImageDataset(data_root, transform=transform)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training
    )
    
    return dataloader


def create_mae_dataloaders(config: dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    根据配置创建MAE训练和验证数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        训练数据加载器和验证数据加载器(可选)
    """
    dataset_config = config.get('dataset', {})
    
    # 获取数据集配置
    dataset_name = dataset_config.get('name', 'cifar10')
    data_root = dataset_config.get('root', './data')
    img_size = config.get('img_size', 224)
    batch_size = config.get('batch_size', 32)
    num_workers = dataset_config.get('num_workers', 4)
    
    # 创建训练数据加载器
    train_loader = create_mae_dataloader(
        dataset_name=dataset_name,
        data_root=data_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        is_training=True
    )
    
    # 创建验证数据加载器(如果需要)
    val_loader = None
    if dataset_config.get('use_validation', False):
        val_batch_size = dataset_config.get('val_batch_size', batch_size)
        val_loader = create_mae_dataloader(
            dataset_name=dataset_name,
            data_root=data_root,
            img_size=img_size,
            batch_size=val_batch_size,
            num_workers=num_workers,
            is_training=False
        )
    
    return train_loader, val_loader


def visualize_mae_reconstruction(model: nn.Module, dataloader: DataLoader, device: torch.device, 
                                num_samples: int = 8, mask_ratio: float = 0.75) -> None:
    """
    可视化MAE重建结果
    
    Args:
        model: MAE模型
        dataloader: 数据加载器
        device: 设备
        num_samples: 样本数量
        mask_ratio: 掩码比例
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            images = images.to(device)
            
            # 前向传播
            loss, pred, mask = model(images, mask_ratio)
            
            # 重建图像
            pred_imgs = model.unpatchify(pred)
            
            # 创建掩码图像
            mask_imgs = images.clone()
            mask_patches = mask.unsqueeze(-1).repeat(1, 1, model.patch_size**2 * 3)
            mask_patches = model.unpatchify(mask_patches)
            mask_imgs = mask_imgs * (1 - mask_patches)
            
            # 显示结果
            fig, axes = plt.subplots(3, min(num_samples, images.size(0)), figsize=(15, 9))
            
            for i in range(min(num_samples, images.size(0))):
                # 原图
                orig_img = images[i].cpu().permute(1, 2, 0)
                orig_img = orig_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                orig_img = torch.clamp(orig_img, 0, 1)
                axes[0, i].imshow(orig_img)
                axes[0, i].set_title('原图')
                axes[0, i].axis('off')
                
                # 掩码图
                masked_img = mask_imgs[i].cpu().permute(1, 2, 0)
                masked_img = masked_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                masked_img = torch.clamp(masked_img, 0, 1)
                axes[1, i].imshow(masked_img)
                axes[1, i].set_title(f'掩码图 ({mask_ratio*100:.0f}%)')
                axes[1, i].axis('off')
                
                # 重建图
                recon_img = pred_imgs[i].cpu().permute(1, 2, 0)
                recon_img = recon_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                recon_img = torch.clamp(recon_img, 0, 1)
                axes[2, i].imshow(recon_img)
                axes[2, i].set_title('重建图')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'mae_reconstruction_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            break


# 数据集信息
DATASET_INFO = {
    'imagenet': {
        'num_classes': 1000,
        'input_size': 224,
        'description': 'ImageNet-1K 大规模图像分类数据集，包含1000个类别'
    },
    'cifar10': {
        'num_classes': 10,
        'input_size': 32,
        'description': 'CIFAR-10 小规模图像分类数据集，包含10个类别'
    },
    'cifar100': {
        'num_classes': 100,
        'input_size': 32,
        'description': 'CIFAR-100 中等规模图像分类数据集，包含100个类别'
    },
    'custom': {
        'num_classes': None,
        'input_size': None,
        'description': '自定义图像数据集'
    }
}


def get_dataset_info(dataset_name: str) -> dict:
    """
    获取数据集信息
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        数据集信息字典
    """
    return DATASET_INFO.get(dataset_name.lower(), DATASET_INFO['custom'])