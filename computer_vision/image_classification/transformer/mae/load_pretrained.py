"""
MAE HuggingFace预训练模型加载器

支持从HuggingFace Hub加载预训练的MAE模型权重
"""

import torch
import torch.nn as nn
from transformers import ViTMAEForPreTraining, ViTMAEConfig
import os
import sys
from typing import Dict, Any, Optional
import requests
import json

# 添加路径以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

from computer_vision.image_classification.transformer.mae.model import MAE
from utils.logger import setup_logger


class HuggingFaceMAELoader:
    """
    HuggingFace MAE模型加载器
    
    支持从HuggingFace Hub加载预训练的MAE模型并转换为我们的实现
    """
    
    def __init__(self):
        self.logger = setup_logger('HF_MAE_Loader')
        
        # 可用的预训练模型
        self.available_models = {
            'mae-base': {
                'model_name': 'facebook/vit-mae-base',
                'description': 'MAE Base模型，在ImageNet-1K上预训练',
                'config': {
                    'img_size': 224,
                    'patch_size': 16,
                    'embed_dim': 768,
                    'depth': 12,
                    'num_heads': 12,
                    'decoder_embed_dim': 512,
                    'decoder_depth': 8,
                    'decoder_num_heads': 16,
                    'mlp_ratio': 4.0,
                    'in_chans': 3,
                    'mask_ratio': 0.75,
                    'norm_pix_loss': True
                }
            },
            'mae-large': {
                'model_name': 'facebook/vit-mae-large',
                'description': 'MAE Large模型，在ImageNet-1K上预训练',
                'config': {
                    'img_size': 224,
                    'patch_size': 16,
                    'embed_dim': 1024,
                    'depth': 24,
                    'num_heads': 16,
                    'decoder_embed_dim': 512,
                    'decoder_depth': 8,
                    'decoder_num_heads': 16,
                    'mlp_ratio': 4.0,
                    'in_chans': 3,
                    'mask_ratio': 0.75,
                    'norm_pix_loss': True
                }
            },
            'mae-huge': {
                'model_name': 'facebook/vit-mae-huge',
                'description': 'MAE Huge模型，在ImageNet-1K上预训练',
                'config': {
                    'img_size': 224,
                    'patch_size': 14,
                    'embed_dim': 1280,
                    'depth': 32,
                    'num_heads': 16,
                    'decoder_embed_dim': 512,
                    'decoder_depth': 8,
                    'decoder_num_heads': 16,
                    'mlp_ratio': 4.0,
                    'in_chans': 3,
                    'mask_ratio': 0.75,
                    'norm_pix_loss': True
                }
            }
        }
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有可用的预训练模型
        
        Returns:
            可用模型字典
        """
        return self.available_models
    
    def load_pretrained_model(self, model_name: str, device: torch.device = None) -> MAE:
        """
        加载预训练的MAE模型
        
        Args:
            model_name: 模型名称 ('mae-base', 'mae-large', 'mae-huge')
            device: 目标设备
            
        Returns:
            加载了预训练权重的MAE模型
        """
        if model_name not in self.available_models:
            raise ValueError(f"不支持的模型: {model_name}. 可用模型: {list(self.available_models.keys())}")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_info = self.available_models[model_name]
        hf_model_name = model_info['model_name']
        config = model_info['config']
        
        self.logger.info(f"正在加载预训练模型: {hf_model_name}")
        
        try:
            # 从HuggingFace加载模型
            hf_model = ViTMAEForPreTraining.from_pretrained(hf_model_name)
            hf_config = hf_model.config
            
            # 创建我们的模型实例
            our_model = MAE(config)
            
            # 转换权重
            self._convert_weights(hf_model, our_model, hf_config)
            
            our_model.to(device)
            our_model.eval()
            
            self.logger.info(f"成功加载预训练模型: {model_name}")
            return our_model
            
        except Exception as e:
            self.logger.error(f"加载预训练模型失败: {e}")
            raise
    
    def _convert_weights(self, hf_model: ViTMAEForPreTraining, our_model: MAE, hf_config: ViTMAEConfig):
        """
        转换HuggingFace模型权重到我们的实现
        
        Args:
            hf_model: HuggingFace模型
            our_model: 我们的模型
            hf_config: HuggingFace配置
        """
        self.logger.info("开始转换模型权重...")
        
        # 获取HuggingFace模型的状态字典
        hf_state_dict = hf_model.state_dict()
        
        # 创建权重映射
        weight_mapping = self._create_weight_mapping(hf_config)
        
        # 转换权重
        our_state_dict = our_model.state_dict()
        converted_count = 0
        
        for our_key, hf_key in weight_mapping.items():
            if hf_key in hf_state_dict and our_key in our_state_dict:
                # 检查形状是否匹配
                hf_weight = hf_state_dict[hf_key]
                our_weight = our_state_dict[our_key]
                
                if hf_weight.shape == our_weight.shape:
                    our_state_dict[our_key] = hf_weight.clone()
                    converted_count += 1
                else:
                    self.logger.warning(f"权重形状不匹配: {our_key} {our_weight.shape} vs {hf_key} {hf_weight.shape}")
        
        # 加载转换后的权重
        our_model.load_state_dict(our_state_dict, strict=False)
        
        self.logger.info(f"权重转换完成，成功转换 {converted_count} 个权重")
    
    def _create_weight_mapping(self, hf_config: ViTMAEConfig) -> Dict[str, str]:
        """
        创建权重映射字典
        
        Args:
            hf_config: HuggingFace配置
            
        Returns:
            权重映射字典 {our_key: hf_key}
        """
        mapping = {}
        
        # Patch embedding
        mapping.update({
            'encoder.patch_embed.proj.weight': 'vit.embeddings.patch_embeddings.projection.weight',
            'encoder.patch_embed.proj.bias': 'vit.embeddings.patch_embeddings.projection.bias',
        })
        
        # Position embeddings
        mapping.update({
            'encoder.pos_embed': 'vit.embeddings.position_embeddings',
            'encoder.cls_token': 'vit.embeddings.cls_token',
        })
        
        # Encoder blocks
        for i in range(hf_config.num_hidden_layers):
            layer_mapping = {
                # Layer norm
                f'encoder.blocks.{i}.norm1.weight': f'vit.encoder.layer.{i}.layernorm_before.weight',
                f'encoder.blocks.{i}.norm1.bias': f'vit.encoder.layer.{i}.layernorm_before.bias',
                f'encoder.blocks.{i}.norm2.weight': f'vit.encoder.layer.{i}.layernorm_after.weight',
                f'encoder.blocks.{i}.norm2.bias': f'vit.encoder.layer.{i}.layernorm_after.bias',
                
                # Attention
                f'encoder.blocks.{i}.attn.qkv.weight': f'vit.encoder.layer.{i}.attention.attention.query.weight',  # 需要特殊处理
                f'encoder.blocks.{i}.attn.proj.weight': f'vit.encoder.layer.{i}.attention.output.dense.weight',
                f'encoder.blocks.{i}.attn.proj.bias': f'vit.encoder.layer.{i}.attention.output.dense.bias',
                
                # MLP
                f'encoder.blocks.{i}.mlp.fc1.weight': f'vit.encoder.layer.{i}.intermediate.dense.weight',
                f'encoder.blocks.{i}.mlp.fc1.bias': f'vit.encoder.layer.{i}.intermediate.dense.bias',
                f'encoder.blocks.{i}.mlp.fc2.weight': f'vit.encoder.layer.{i}.output.dense.weight',
                f'encoder.blocks.{i}.mlp.fc2.bias': f'vit.encoder.layer.{i}.output.dense.bias',
            }
            mapping.update(layer_mapping)
        
        # Encoder layer norm
        mapping.update({
            'encoder.norm.weight': 'vit.layernorm.weight',
            'encoder.norm.bias': 'vit.layernorm.bias',
        })
        
        # Decoder
        mapping.update({
            'decoder.decoder_embed.weight': 'decoder.decoder_embed.weight',
            'decoder.decoder_embed.bias': 'decoder.decoder_embed.bias',
            'decoder.mask_token': 'decoder.mask_token',
            'decoder.decoder_pos_embed': 'decoder.decoder_pos_embed',
        })
        
        # Decoder blocks
        for i in range(hf_config.decoder_num_hidden_layers):
            layer_mapping = {
                f'decoder.decoder_blocks.{i}.norm1.weight': f'decoder.decoder_layers.{i}.layernorm_before.weight',
                f'decoder.decoder_blocks.{i}.norm1.bias': f'decoder.decoder_layers.{i}.layernorm_before.bias',
                f'decoder.decoder_blocks.{i}.norm2.weight': f'decoder.decoder_layers.{i}.layernorm_after.weight',
                f'decoder.decoder_blocks.{i}.norm2.bias': f'decoder.decoder_layers.{i}.layernorm_after.bias',
                
                f'decoder.decoder_blocks.{i}.mlp.fc1.weight': f'decoder.decoder_layers.{i}.intermediate.dense.weight',
                f'decoder.decoder_blocks.{i}.mlp.fc1.bias': f'decoder.decoder_layers.{i}.intermediate.dense.bias',
                f'decoder.decoder_blocks.{i}.mlp.fc2.weight': f'decoder.decoder_layers.{i}.output.dense.weight',
                f'decoder.decoder_blocks.{i}.mlp.fc2.bias': f'decoder.decoder_layers.{i}.output.dense.bias',
            }
            mapping.update(layer_mapping)
        
        # Decoder output
        mapping.update({
            'decoder.decoder_norm.weight': 'decoder.decoder_norm.weight',
            'decoder.decoder_norm.bias': 'decoder.decoder_norm.bias',
            'decoder.decoder_pred.weight': 'decoder.decoder_pred.weight',
            'decoder.decoder_pred.bias': 'decoder.decoder_pred.bias',
        })
        
        return mapping
    
    def save_converted_model(self, model: MAE, save_path: str, config: Dict[str, Any]):
        """
        保存转换后的模型
        
        Args:
            model: 转换后的模型
            save_path: 保存路径
            config: 模型配置
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'model_type': 'mae',
            'source': 'huggingface_converted'
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"模型已保存到: {save_path}")
    
    def download_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        下载模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息
        """
        if model_name not in self.available_models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        hf_model_name = self.available_models[model_name]['model_name']
        
        try:
            # 获取模型信息
            api_url = f"https://huggingface.co/api/models/{hf_model_name}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                model_info = response.json()
                return {
                    'model_name': model_name,
                    'hf_model_name': hf_model_name,
                    'downloads': model_info.get('downloads', 0),
                    'likes': model_info.get('likes', 0),
                    'tags': model_info.get('tags', []),
                    'description': self.available_models[model_name]['description'],
                    'config': self.available_models[model_name]['config']
                }
            else:
                self.logger.warning(f"无法获取模型信息: {response.status_code}")
                return self.available_models[model_name]
                
        except Exception as e:
            self.logger.warning(f"下载模型信息失败: {e}")
            return self.available_models[model_name]


def load_pretrained_mae(model_name: str = 'mae-base', device: torch.device = None, 
                       save_path: Optional[str] = None) -> MAE:
    """
    便捷函数：加载预训练的MAE模型
    
    Args:
        model_name: 模型名称 ('mae-base', 'mae-large', 'mae-huge')
        device: 目标设备
        save_path: 可选的保存路径
        
    Returns:
        加载了预训练权重的MAE模型
    """
    loader = HuggingFaceMAELoader()
    model = loader.load_pretrained_model(model_name, device)
    
    if save_path:
        config = loader.available_models[model_name]['config']
        loader.save_converted_model(model, save_path, config)
    
    return model


def list_pretrained_models() -> None:
    """列出所有可用的预训练模型"""
    loader = HuggingFaceMAELoader()
    models = loader.list_available_models()
    
    print("可用的预训练MAE模型:")
    print("=" * 50)
    
    for name, info in models.items():
        print(f"模型名称: {name}")
        print(f"HuggingFace模型: {info['model_name']}")
        print(f"描述: {info['description']}")
        config = info['config']
        print(f"配置: 图像尺寸={config['img_size']}, 嵌入维度={config['embed_dim']}, 层数={config['depth']}")
        print("-" * 50)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='加载HuggingFace预训练MAE模型')
    parser.add_argument('--model', type=str, default='mae-base', 
                       choices=['mae-base', 'mae-large', 'mae-huge'],
                       help='预训练模型名称')
    parser.add_argument('--save', type=str, default=None, help='保存转换后模型的路径')
    parser.add_argument('--list', action='store_true', help='列出所有可用模型')
    parser.add_argument('--info', action='store_true', help='显示模型信息')
    
    args = parser.parse_args()
    
    if args.list:
        list_pretrained_models()
    elif args.info:
        loader = HuggingFaceMAELoader()
        info = loader.download_model_info(args.model)
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_pretrained_mae(args.model, device, args.save)
        
        print(f"成功加载预训练模型: {args.model}")
        print(f"模型参数数量: {model.get_num_params():,}")
        
        # 测试前向传播
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            loss, pred, mask = model(test_input)
        print(f"测试前向传播成功 - 损失: {loss.item():.4f}")