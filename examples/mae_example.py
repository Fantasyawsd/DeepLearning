"""
Example usage of MAE (Masked Autoencoder) model.
"""

import torch
import numpy as np
from models import MAE
from utils import Config

def mae_example():
    """Example usage of MAE model."""
    print("=== MAE (Masked Autoencoder) Example ===")
    
    # Create configuration
    config = Config({
        'model_name': 'mae',
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'embed_dim': 768,
        'encoder_depth': 12,
        'encoder_num_heads': 12,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'mlp_ratio': 4.0,
        'mask_ratio': 0.75,
        'norm_pix_loss': True
    })
    
    # Create model
    model = MAE(config.to_dict())
    model.eval()
    
    print(f"Model: {model.model_name}")
    model.summary()
    
    # Create dummy input (batch of images)
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput shape: {images.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
    
    print(f"\nOutput keys: {outputs.keys()}")
    print(f"Loss: {outputs['loss'].item():.6f}")
    print(f"Predictions shape: {outputs['pred'].shape}")
    print(f"Mask shape: {outputs['mask'].shape}")
    print(f"Latent shape: {outputs['latent'].shape}")
    
    # Demonstrate reconstruction
    print(f"\nMask ratio: {config.get('mask_ratio', 0.75) * 100:.1f}%")
    print(f"Number of masked patches: {outputs['mask'].sum().item():.0f}")
    print(f"Total patches: {outputs['mask'].numel()}")
    
    print("\n=== MAE Example Completed ===\n")

if __name__ == '__main__':
    mae_example()