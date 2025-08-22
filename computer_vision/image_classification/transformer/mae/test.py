"""
MAE模型测试脚本

支持模型推理、重建可视化、特征提取等功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import sys

# 添加路径以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

from computer_vision.image_classification.transformer.mae.model import MAE
from computer_vision.image_classification.transformer.mae.dataset import create_mae_dataloader, get_mae_transforms
from utils.logger import setup_logger
from utils.config import Config


class MAETester:
    """
    MAE模型测试器
    
    支持模型推理、重建可视化、特征提取等功能
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str, device: torch.device):
        """
        初始化测试器
        
        Args:
            config: 测试配置
            model_path: 模型权重路径
            device: 测试设备
        """
        self.config = config
        self.device = device
        self.model_path = model_path
        
        # 设置日志
        self.logger = setup_logger('MAE_Testing')
        
        # 创建输出目录
        self.output_dir = config.get('output_dir', './outputs/mae_test')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 加载模型权重
        self._load_model()
    
    def _init_model(self):
        """初始化模型"""
        self.model = MAE(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"模型参数数量: {self.model.get_num_params():,}")
    
    def _load_model(self):
        """加载模型权重"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理DDP模型的state_dict
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.logger.info(f"成功加载模型权重: {self.model_path}")
    
    def test_reconstruction(self, dataloader: DataLoader, num_samples: int = 8, 
                          mask_ratio: float = 0.75, save_images: bool = True) -> Dict[str, float]:
        """
        测试图像重建能力
        
        Args:
            dataloader: 数据加载器
            num_samples: 测试样本数量
            mask_ratio: 掩码比例
            save_images: 是否保存重建图像
            
        Returns:
            测试结果统计
        """
        self.logger.info(f"开始测试图像重建，掩码比例: {mask_ratio:.1%}")
        
        self.model.eval()
        total_loss = 0.0
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc='Testing')):
                if batch_idx >= num_samples // dataloader.batch_size + 1:
                    break
                
                images = images.to(self.device)
                
                # 前向传播
                loss, pred, mask = self.model(images, mask_ratio)
                total_loss += loss.item()
                
                # 计算重建误差
                pred_imgs = self.model.unpatchify(pred)
                mse_error = nn.MSELoss()(pred_imgs, images).item()
                reconstruction_errors.append(mse_error)
                
                # 保存重建图像
                if save_images and batch_idx < 5:  # 只保存前5个批次
                    self._save_reconstruction_images(images, pred_imgs, mask, batch_idx, mask_ratio)
        
        # 计算统计结果
        avg_loss = total_loss / min(len(dataloader), num_samples // dataloader.batch_size + 1)
        avg_mse = np.mean(reconstruction_errors)
        std_mse = np.std(reconstruction_errors)
        
        results = {
            'avg_reconstruction_loss': avg_loss,
            'avg_mse_error': avg_mse,
            'std_mse_error': std_mse,
            'mask_ratio': mask_ratio,
            'num_samples_tested': len(reconstruction_errors) * dataloader.batch_size
        }
        
        self.logger.info(f"重建测试完成 - 平均损失: {avg_loss:.4f}, 平均MSE: {avg_mse:.4f}")
        return results
    
    def _save_reconstruction_images(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                                  mask: torch.Tensor, batch_idx: int, mask_ratio: float):
        """保存重建图像对比"""
        batch_size = min(8, original.size(0))
        
        # 创建掩码图像
        mask_patches = mask.unsqueeze(-1).repeat(1, 1, self.model.patch_size**2 * 3)
        mask_patches = self.model.unpatchify(mask_patches)
        masked_imgs = original * (1 - mask_patches)
        
        # 准备显示
        fig, axes = plt.subplots(3, batch_size, figsize=(batch_size * 3, 9))
        if batch_size == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(batch_size):
            # 反标准化
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            
            # 原图
            orig_img = original[i].cpu().permute(1, 2, 0)
            orig_img = orig_img * std + mean
            orig_img = torch.clamp(orig_img, 0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title('原图')
            axes[0, i].axis('off')
            
            # 掩码图
            masked_img = masked_imgs[i].cpu().permute(1, 2, 0)
            masked_img = masked_img * std + mean
            masked_img = torch.clamp(masked_img, 0, 1)
            axes[1, i].imshow(masked_img)
            axes[1, i].set_title(f'掩码图 ({mask_ratio*100:.0f}%)')
            axes[1, i].axis('off')
            
            # 重建图
            recon_img = reconstructed[i].cpu().permute(1, 2, 0)
            recon_img = recon_img * std + mean
            recon_img = torch.clamp(recon_img, 0, 1)
            axes[2, i].imshow(recon_img)
            axes[2, i].set_title('重建图')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'reconstruction_batch_{batch_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_different_mask_ratios(self, dataloader: DataLoader, 
                                 mask_ratios: List[float] = [0.25, 0.5, 0.75, 0.9]) -> Dict[str, Any]:
        """
        测试不同掩码比例的重建效果
        
        Args:
            dataloader: 数据加载器
            mask_ratios: 掩码比例列表
            
        Returns:
            不同掩码比例的测试结果
        """
        self.logger.info("测试不同掩码比例的重建效果")
        
        results = {}
        
        for mask_ratio in mask_ratios:
            self.logger.info(f"测试掩码比例: {mask_ratio:.1%}")
            result = self.test_reconstruction(dataloader, num_samples=32, 
                                            mask_ratio=mask_ratio, save_images=False)
            results[f'mask_{mask_ratio:.2f}'] = result
        
        # 绘制结果对比
        self._plot_mask_ratio_comparison(results)
        
        return results
    
    def _plot_mask_ratio_comparison(self, results: Dict[str, Dict[str, float]]):
        """绘制掩码比例对比图"""
        mask_ratios = []
        losses = []
        mse_errors = []
        
        for key, result in results.items():
            mask_ratio = result['mask_ratio']
            mask_ratios.append(mask_ratio)
            losses.append(result['avg_reconstruction_loss'])
            mse_errors.append(result['avg_mse_error'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 重建损失
        ax1.plot(mask_ratios, losses, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('掩码比例')
        ax1.set_ylabel('重建损失')
        ax1.set_title('重建损失 vs 掩码比例')
        ax1.grid(True, alpha=0.3)
        
        # MSE误差
        ax2.plot(mask_ratios, mse_errors, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('掩码比例')
        ax2.set_ylabel('MSE误差')
        ax2.set_title('MSE误差 vs 掩码比例')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mask_ratio_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def extract_features(self, dataloader: DataLoader, layer: str = 'encoder') -> np.ndarray:
        """
        提取图像特征
        
        Args:
            dataloader: 数据加载器
            layer: 提取特征的层 ('encoder', 'decoder')
            
        Returns:
            提取的特征
        """
        self.logger.info(f"开始提取{layer}特征")
        
        features = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc='Extracting features'):
                images = images.to(self.device)
                
                if layer == 'encoder':
                    # 使用编码器提取特征
                    latent, _, _ = self.model.encoder(images, mask_ratio=0.0)  # 不掩码
                    feat = latent[:, 0]  # CLS token
                elif layer == 'decoder':
                    # 使用解码器提取特征
                    latent, _, ids_restore = self.model.encoder(images, mask_ratio=0.0)
                    feat = self.model.decoder(latent, ids_restore)
                    feat = feat.mean(dim=1)  # 平均pooling
                else:
                    raise ValueError(f"不支持的层: {layer}")
                
                features.append(feat.cpu().numpy())
                labels.append(targets.numpy())
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        self.logger.info(f"特征提取完成，特征形状: {features.shape}")
        
        # 保存特征
        np.save(os.path.join(self.output_dir, f'{layer}_features.npy'), features)
        np.save(os.path.join(self.output_dir, f'{layer}_labels.npy'), labels)
        
        return features
    
    def test_single_image(self, image_path: str, mask_ratio: float = 0.75) -> Dict[str, Any]:
        """
        测试单张图像
        
        Args:
            image_path: 图像路径
            mask_ratio: 掩码比例
            
        Returns:
            测试结果
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 加载和预处理图像
        transform = get_mae_transforms(self.config.get('img_size', 224), is_training=False)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            loss, pred, mask = self.model(image_tensor, mask_ratio)
        
        # 重建图像
        pred_img = self.model.unpatchify(pred)
        
        # 创建掩码图像
        mask_patches = mask.unsqueeze(-1).repeat(1, 1, self.model.patch_size**2 * 3)
        mask_patches = self.model.unpatchify(mask_patches)
        masked_img = image_tensor * (1 - mask_patches)
        
        # 可视化结果
        self._visualize_single_image_result(image_tensor, masked_img, pred_img, mask_ratio, image_path)
        
        return {
            'reconstruction_loss': loss.item(),
            'mask_ratio': mask_ratio,
            'image_path': image_path
        }
    
    def _visualize_single_image_result(self, original: torch.Tensor, masked: torch.Tensor, 
                                     reconstructed: torch.Tensor, mask_ratio: float, image_path: str):
        """可视化单张图像的测试结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 反标准化
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        # 原图
        orig_img = original[0].cpu().permute(1, 2, 0)
        orig_img = orig_img * std + mean
        orig_img = torch.clamp(orig_img, 0, 1)
        axes[0].imshow(orig_img)
        axes[0].set_title('原图')
        axes[0].axis('off')
        
        # 掩码图
        masked_img = masked[0].cpu().permute(1, 2, 0)
        masked_img = masked_img * std + mean
        masked_img = torch.clamp(masked_img, 0, 1)
        axes[1].imshow(masked_img)
        axes[1].set_title(f'掩码图 ({mask_ratio*100:.0f}%)')
        axes[1].axis('off')
        
        # 重建图
        recon_img = reconstructed[0].cpu().permute(1, 2, 0)
        recon_img = recon_img * std + mean
        recon_img = torch.clamp(recon_img, 0, 1)
        axes[2].imshow(recon_img)
        axes[2].set_title('重建图')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        filename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(self.output_dir, f'single_test_{filename}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def benchmark_inference(self, dataloader: DataLoader, num_batches: int = 100) -> Dict[str, float]:
        """
        基准测试推理性能
        
        Args:
            dataloader: 数据加载器
            num_batches: 测试批次数
            
        Returns:
            性能统计
        """
        self.logger.info("开始推理性能基准测试")
        
        self.model.eval()
        inference_times = []
        
        # 预热
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= 5:  # 预热5个批次
                    break
                images = images.to(self.device)
                _ = self.model(images, mask_ratio=0.75)
        
        # 正式测试
        with torch.no_grad():
            for i, (images, _) in enumerate(tqdm(dataloader, desc='Benchmarking')):
                if i >= num_batches:
                    break
                
                images = images.to(self.device)
                
                start_time = time.time()
                _ = self.model(images, mask_ratio=0.75)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
        
        # 计算统计
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        throughput = dataloader.batch_size / avg_time
        
        results = {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'throughput_samples_per_sec': throughput,
            'batch_size': dataloader.batch_size,
            'num_batches_tested': len(inference_times)
        }
        
        self.logger.info(f"推理性能测试完成 - 平均时间: {avg_time:.4f}s, 吞吐量: {throughput:.2f} samples/s")
        return results


def test_mae(config_path: str, model_path: str, test_type: str = 'reconstruction'):
    """
    测试MAE模型的主函数
    
    Args:
        config_path: 配置文件路径
        model_path: 模型权重路径
        test_type: 测试类型 ('reconstruction', 'features', 'benchmark', 'single')
    """
    # 加载配置
    config = Config.from_file(config_path)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试器
    tester = MAETester(config.to_dict(), model_path, device)
    
    # 创建数据加载器
    test_loader = create_mae_dataloader(
        dataset_name=config.dataset.name,
        data_root=config.dataset.root,
        img_size=config.img_size,
        batch_size=config.get('test_batch_size', 32),
        num_workers=config.dataset.get('num_workers', 4),
        is_training=False
    )
    
    # 执行不同类型的测试
    results = {}
    
    if test_type == 'reconstruction' or test_type == 'all':
        # 重建测试
        results['reconstruction'] = tester.test_reconstruction(test_loader)
        results['mask_ratios'] = tester.test_different_mask_ratios(test_loader)
    
    if test_type == 'features' or test_type == 'all':
        # 特征提取
        results['features'] = {
            'encoder_features_shape': tester.extract_features(test_loader, 'encoder').shape
        }
    
    if test_type == 'benchmark' or test_type == 'all':
        # 性能基准测试
        results['benchmark'] = tester.benchmark_inference(test_loader)
    
    # 保存测试结果
    results_path = os.path.join(tester.output_dir, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"测试完成！结果保存到: {results_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试MAE模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--test_type', type=str, default='reconstruction', 
                       choices=['reconstruction', 'features', 'benchmark', 'single', 'all'],
                       help='测试类型')
    parser.add_argument('--image', type=str, default=None, help='单张图像路径(用于single测试)')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 单张图像测试
    if args.test_type == 'single':
        if not args.image:
            raise ValueError("单张图像测试需要指定--image参数")
        
        config = Config.from_file(args.config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.output_dir:
            config.output_dir = args.output_dir
        
        tester = MAETester(config.to_dict(), args.model, device)
        result = tester.test_single_image(args.image)
        print(f"单张图像测试完成: {result}")
    else:
        # 批量测试
        if args.output_dir:
            config = Config.from_file(args.config)
            config.output_dir = args.output_dir
            config.save(args.config)
        
        test_mae(args.config, args.model, args.test_type)