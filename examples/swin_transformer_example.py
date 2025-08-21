"""
Example usage of Swin Transformer model.
"""

import torch
from models import SwinTransformer
from utils import Config

def swin_transformer_example():
    """Example usage of Swin Transformer model."""
    print("=== Swin Transformer Example ===")
    
    # Create configuration
    config = Config({
        'model_name': 'swin_transformer',
        'img_size': 224,
        'patch_size': 4,
        'in_chans': 3,
        'num_classes': 1000,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'ape': False,
        'patch_norm': True
    })
    
    # Create model
    model = SwinTransformer(config.to_dict())
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
    
    print(f"Output shape: {outputs.shape}")
    print(f"Number of classes: {config.get('num_classes', 1000)}")
    
    # Show model architecture details
    print(f"\nModel Architecture Details:")
    print(f"  Image size: {config.get('img_size')}x{config.get('img_size')}")
    print(f"  Patch size: {config.get('patch_size')}x{config.get('patch_size')}")
    print(f"  Embed dim: {config.get('embed_dim')}")
    print(f"  Depths: {config.get('depths')}")
    print(f"  Num heads: {config.get('num_heads')}")
    print(f"  Window size: {config.get('window_size')}")
    
    # Calculate patches resolution
    patch_size = config.get('patch_size', 4)
    img_size = config.get('img_size', 224)
    patches_resolution = img_size // patch_size
    print(f"  Patches resolution: {patches_resolution}x{patches_resolution}")
    print(f"  Total patches: {patches_resolution * patches_resolution}")
    
    # Show feature map sizes at each stage
    embed_dim = config.get('embed_dim', 96)
    depths = config.get('depths', [2, 2, 6, 2])
    print(f"\nFeature map sizes at each stage:")
    current_res = patches_resolution
    current_dim = embed_dim
    for i, depth in enumerate(depths):
        print(f"  Stage {i+1}: {current_res}x{current_res}, {current_dim} channels, {depth} blocks")
        if i < len(depths) - 1:  # Not the last stage
            current_res //= 2
            current_dim *= 2
    
    print("\n=== Swin Transformer Example Completed ===\n")

if __name__ == '__main__':
    swin_transformer_example()