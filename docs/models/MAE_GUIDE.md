# MAE (Masked Autoencoder) - Complete Guide

## Overview

MAE (Masked Autoencoder) is a self-supervised learning method for computer vision introduced in the paper "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022). It learns visual representations by reconstructing masked portions of input images.

## Key Concepts

### Architecture
- **Encoder**: Vision Transformer (ViT) that processes only visible patches
- **Decoder**: Lightweight transformer that reconstructs the full image from latent representation
- **Masking**: Randomly masks 75% of image patches during training

### How It Works
1. Divide image into patches
2. Randomly mask 75% of patches
3. Encode only visible patches with ViT encoder
4. Add mask tokens and positional embeddings
5. Decode to reconstruct the original image
6. Compute reconstruction loss only on masked patches

## Configuration

### Configuration File: `configs/mae_config.yaml`

```yaml
model_name: "mae"
img_size: 224          # Input image size
patch_size: 16         # Patch size for tokenization
in_chans: 3           # Input channels (RGB)
embed_dim: 768        # Encoder embedding dimension
encoder_depth: 12     # Number of encoder layers
encoder_num_heads: 12 # Number of attention heads in encoder
decoder_embed_dim: 512 # Decoder embedding dimension
decoder_depth: 8     # Number of decoder layers
decoder_num_heads: 16 # Number of attention heads in decoder
mlp_ratio: 4.0       # MLP expansion ratio
mask_ratio: 0.75     # Fraction of patches to mask
norm_pix_loss: true  # Whether to normalize pixel values in loss
```

### Key Parameters

- **`mask_ratio`**: Controls how much of the image is masked (0.75 = 75%)
- **`norm_pix_loss`**: Whether to normalize pixel values for loss computation
- **`embed_dim`**: Controls model capacity (768 for base, 1024 for large)
- **`patch_size`**: Smaller patches = higher resolution but more computation

## Usage Examples

### Basic Usage

```python
import torch
from models import MAE
from utils import Config

# Load configuration
config = Config.from_file('configs/mae_config.yaml')

# Create model
model = MAE(config.to_dict())
model.eval()

# Prepare input (batch of RGB images)
batch_size = 4
images = torch.randn(batch_size, 3, 224, 224)

# Forward pass
with torch.no_grad():
    outputs = model(images)

print(f"Reconstruction loss: {outputs['loss'].item():.6f}")
print(f"Predictions shape: {outputs['pred'].shape}")  # Reconstructed patches
print(f"Mask shape: {outputs['mask'].shape}")         # Binary mask
print(f"Latent shape: {outputs['latent'].shape}")     # Encoded features
```

### Custom Configuration

```python
from models import MAE
from utils import Config

# Custom configuration for smaller model
config = Config({
    'model_name': 'mae',
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 384,        # Smaller embedding dimension
    'encoder_depth': 6,      # Fewer encoder layers
    'encoder_num_heads': 6,  # Fewer attention heads
    'decoder_embed_dim': 256,
    'decoder_depth': 4,
    'decoder_num_heads': 8,
    'mask_ratio': 0.75,
    'norm_pix_loss': True
})

model = MAE(config.to_dict())
print(f"Model parameters: {model.count_parameters():,}")
```

### Visualization of Masking

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from models import MAE
from utils import Config

def visualize_mae_reconstruction(model, image):
    """Visualize original, masked, and reconstructed images."""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        outputs = model(image)
        
        # Get outputs
        pred = outputs['pred']      # Reconstructed patches
        mask = outputs['mask']      # Binary mask
        
        # Reconstruct full image
        reconstructed = model.unpatchify(pred)
        
        # Create masked image for visualization
        masked_img = image.clone()
        patch_size = model.patch_embed.patch_size[0]
        H = W = image.shape[-1] // patch_size
        
        mask_2d = mask.reshape(-1, H, W).unsqueeze(1)
        mask_img = torch.nn.functional.interpolate(
            mask_2d.float(), size=(image.shape[-1], image.shape[-1]), 
            mode='nearest'
        )
        
        # Apply mask (masked regions become gray)
        masked_img = image * (1 - mask_img) + 0.5 * mask_img
        
        return image[0], masked_img[0], reconstructed[0]

# Example usage
config = Config.from_file('configs/mae_config.yaml')
model = MAE(config.to_dict())

# Create sample image
image = torch.randn(3, 224, 224)

# Visualize
original, masked, reconstructed = visualize_mae_reconstruction(model, image)
```

## Training

### Self-Supervised Pre-training

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import MAE
from utils import Config

# Configuration
config = Config.from_file('configs/mae_config.yaml')
model = MAE(config.to_dict())

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
    
    # Save checkpoint
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'mae_checkpoint_epoch_{epoch}.pth')
```

### Fine-tuning for Downstream Tasks

After pre-training, you can fine-tune MAE for specific tasks:

```python
# Load pre-trained MAE
pretrained_mae = MAE(config.to_dict())
pretrained_mae.load_checkpoint('mae_pretrained.pth')

# Create classifier using MAE encoder
class MAEClassifier(nn.Module):
    def __init__(self, mae_model, num_classes):
        super().__init__()
        self.encoder = mae_model.patch_embed
        self.pos_embed = mae_model.pos_embed
        self.blocks = mae_model.blocks
        self.norm = mae_model.norm
        self.cls_head = nn.Linear(mae_model.embed_dim, num_classes)
    
    def forward(self, x):
        # Encode without masking
        x = self.encoder(x)
        x = x + self.pos_embed[:, 1:, :]  # No cls token
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return self.cls_head(x)

# Create classifier
classifier = MAEClassifier(pretrained_mae, num_classes=1000)

# Fine-tune on labeled data
# ... training code for classification task
```

## Advanced Usage

### Custom Masking Strategies

```python
class CustomMAE(MAE):
    def random_masking(self, x, mask_ratio):
        """Custom masking strategy - block masking instead of random."""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # Create block mask instead of random mask
        noise = torch.rand(N, L, device=x.device)
        
        # Create blocks of consecutive masked regions
        block_size = 4  # Mask in 2x2 blocks
        H = W = int(L ** 0.5)
        noise = noise.reshape(N, H, W)
        
        # Apply block masking logic here
        # ... custom masking implementation
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, 
                               index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

# Use custom MAE
custom_model = CustomMAE(config.to_dict())
```

### Multi-Scale Training

```python
def multi_scale_mae_training(model, dataloader, scales=[224, 256, 288]):
    """Train MAE with multiple input scales."""
    model.train()
    
    for epoch in range(num_epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            # Randomly choose scale
            scale = np.random.choice(scales)
            
            # Resize images
            if scale != images.shape[-1]:
                images = torch.nn.functional.interpolate(
                    images, size=(scale, scale), mode='bilinear'
                )
            
            # Forward pass
            outputs = model(images)
            loss = outputs['loss']
            
            # ... rest of training code
```

## Model Variants

### Different Model Sizes

```python
# MAE Base
mae_base_config = {
    'embed_dim': 768,
    'encoder_depth': 12,
    'encoder_num_heads': 12,
    'decoder_embed_dim': 512,
    'decoder_depth': 8,
    'decoder_num_heads': 16
}

# MAE Large
mae_large_config = {
    'embed_dim': 1024,
    'encoder_depth': 24,
    'encoder_num_heads': 16,
    'decoder_embed_dim': 512,
    'decoder_depth': 8,
    'decoder_num_heads': 16
}

# MAE Huge
mae_huge_config = {
    'embed_dim': 1280,
    'encoder_depth': 32,
    'encoder_num_heads': 16,
    'decoder_embed_dim': 512,
    'decoder_depth': 8,
    'decoder_num_heads': 16
}
```

## Tips and Best Practices

### Training Tips

1. **Masking Ratio**: 75% works well for most cases, but try 50-90% for different datasets
2. **Learning Rate**: Start with 1e-4, use warmup and cosine decay
3. **Batch Size**: Use larger batches (256-1024) for better stability
4. **Data Augmentation**: Use minimal augmentation (random crop, flip)

### Performance Optimization

1. **Mixed Precision**: Use automatic mixed precision for faster training
2. **Gradient Checkpointing**: For large models to save memory
3. **Efficient Data Loading**: Use multiple workers and pin_memory=True

### Common Issues

1. **NaN Loss**: Reduce learning rate or check input normalization
2. **Poor Reconstruction**: Verify masking ratio and decoder capacity
3. **Memory Issues**: Reduce batch size or use gradient checkpointing

## Paper Reference

- **Title**: "Masked Autoencoders Are Scalable Vision Learners"
- **Authors**: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick
- **Conference**: CVPR 2022
- **ArXiv**: https://arxiv.org/abs/2111.06377
- **Code**: https://github.com/facebookresearch/mae