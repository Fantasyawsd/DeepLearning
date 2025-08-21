# Swin Transformer - 完整指南

## 概述

Swin Transformer（移位窗口Transformer）是一个分层视觉transformer，在计算机视觉任务上取得了优异性能。在"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"（ICCV 2021）中介绍，通过使用移位窗口和分层特征图解决了将transformer应用于视觉的挑战。

## 核心概念

### 架构
- **分层设计**：像CNN一样产生多尺度特征图
- **移位窗口注意力**：在局部窗口内高效计算注意力
- **补丁合并**：减少空间分辨率同时增加通道维度
- **相对位置偏置**：增强模型理解空间关系的能力

### 关键创新
1. **基于窗口的注意力**：将自注意力限制在局部窗口内以提高效率
2. **移位窗口**：在保持效率的同时实现跨窗口连接
3. **分层特征图**：用于各种视觉任务的多尺度表示
4. **线性计算复杂度**：与图像大小呈线性关系

## 配置

### 配置文件：`configs/swin_config.yaml`

```yaml
model_name: "swin_transformer"
img_size: 224                    # 输入图像大小
patch_size: 4                    # 初始嵌入的补丁大小
in_chans: 3                      # 输入通道数（RGB）
num_classes: 1000                # 输出类别数
embed_dim: 96                    # 初始嵌入维度
depths: [2, 2, 6, 2]            # 每个阶段的块数
num_heads: [3, 6, 12, 24]       # 每个阶段的注意力头数
window_size: 7                   # 注意力窗口大小
mlp_ratio: 4.0                   # MLP扩展比例
qkv_bias: true                   # QKV投影是否使用偏置
qk_scale: null                   # QK点积的缩放因子
drop_rate: 0.0                   # Dropout率
attn_drop_rate: 0.0             # 注意力dropout率
drop_path_rate: 0.1             # 随机深度率
ape: false                      # 绝对位置嵌入
patch_norm: true                # 是否标准化补丁嵌入
```

### 关键参数

- **`depths`**：每个阶段的transformer块数 [阶段1, 阶段2, 阶段3, 阶段4]
- **`num_heads`**：每个阶段的注意力头数
- **`window_size`**：注意力窗口大小（通常为7）
- **`embed_dim`**：起始嵌入维度（每个阶段翻倍）
- **`drop_path_rate`**: Stochastic depth for regularization

## Usage Examples

### Basic Usage

```python
import torch
from models import SwinTransformer
from utils import Config

# Load configuration
config = Config.from_file('configs/swin_config.yaml')

# Create model
model = SwinTransformer(config.to_dict())
model.eval()

# Prepare input (batch of RGB images)
batch_size = 4
images = torch.randn(batch_size, 3, 224, 224)

# Forward pass
with torch.no_grad():
    outputs = model(images)

print(f"Output shape: {outputs.shape}")  # [batch_size, num_classes]
print(f"Model parameters: {model.count_parameters():,}")
```

### Custom Configuration

```python
from models import SwinTransformer
from utils import Config

# Custom configuration for different model sizes
swin_tiny_config = Config({
    'model_name': 'swin_transformer',
    'img_size': 224,
    'patch_size': 4,
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 7,
    'num_classes': 1000
})

swin_small_config = Config({
    'model_name': 'swin_transformer',
    'img_size': 224,
    'patch_size': 4,
    'embed_dim': 96,
    'depths': [2, 2, 18, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 7,
    'num_classes': 1000
})

swin_base_config = Config({
    'model_name': 'swin_transformer',
    'img_size': 224,
    'patch_size': 4,
    'embed_dim': 128,
    'depths': [2, 2, 18, 2],
    'num_heads': [4, 8, 16, 32],
    'window_size': 7,
    'num_classes': 1000
})

# Create models
swin_tiny = SwinTransformer(swin_tiny_config.to_dict())
swin_small = SwinTransformer(swin_small_config.to_dict())
swin_base = SwinTransformer(swin_base_config.to_dict())

print(f"Swin-T parameters: {swin_tiny.count_parameters():,}")
print(f"Swin-S parameters: {swin_small.count_parameters():,}")
print(f"Swin-B parameters: {swin_base.count_parameters():,}")
```

### Feature Extraction

```python
class SwinFeatureExtractor(nn.Module):
    """Extract features from different stages of Swin Transformer."""
    
    def __init__(self, config):
        super().__init__()
        self.swin = SwinTransformer(config)
        # Remove classification head
        self.swin.head = nn.Identity()
    
    def forward(self, x):
        # Get features from all stages
        features = []
        
        x = self.swin.patch_embed(x)
        if self.swin.ape:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.pos_drop(x)
        
        for i, layer in enumerate(self.swin.layers):
            x = layer(x)
            # Store feature maps from each stage
            features.append(x)
        
        x = self.swin.norm(x)  # B L C
        features.append(x)
        
        return features

# Usage
config = Config.from_file('configs/swin_config.yaml')
feature_extractor = SwinFeatureExtractor(config.to_dict())

images = torch.randn(2, 3, 224, 224)
features = feature_extractor(images)

for i, feat in enumerate(features):
    print(f"Stage {i} features shape: {feat.shape}")
```

### Multi-Scale Feature Pyramid

```python
class SwinFPN(nn.Module):
    """Feature Pyramid Network with Swin Transformer backbone."""
    
    def __init__(self, config):
        super().__init__()
        self.backbone = SwinFeatureExtractor(config)
        
        # FPN components
        embed_dim = config['embed_dim']
        self.fpn_dims = [embed_dim * (2**i) for i in range(4)]
        self.output_dim = 256
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, self.output_dim, 1) for dim in self.fpn_dims
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(self.output_dim, self.output_dim, 3, padding=1) for _ in range(4)
        ])
    
    def forward(self, x):
        # Get backbone features
        backbone_features = self.backbone(x)
        
        # Convert to spatial format and apply FPN
        fpn_features = []
        
        for i, (feat, lateral_conv, fpn_conv) in enumerate(zip(
            backbone_features[:4], self.lateral_convs, self.fpn_convs
        )):
            # Reshape from (B, L, C) to (B, C, H, W)
            B, L, C = feat.shape
            H = W = int(L ** 0.5)
            feat_2d = feat.transpose(1, 2).reshape(B, C, H, W)
            
            # Apply lateral connection
            lateral_feat = lateral_conv(feat_2d)
            
            # Upsample and add previous level (if exists)
            if i > 0:
                prev_feat = nn.functional.interpolate(
                    fpn_features[-1], size=lateral_feat.shape[-2:], mode='nearest'
                )
                lateral_feat = lateral_feat + prev_feat
            
            # Apply FPN convolution
            fpn_feat = fpn_conv(lateral_feat)
            fpn_features.append(fpn_feat)
        
        return fpn_features

# Usage for object detection/segmentation
config = Config.from_file('configs/swin_config.yaml')
fpn_model = SwinFPN(config.to_dict())

images = torch.randn(2, 3, 224, 224)
fpn_features = fpn_model(images)

for i, feat in enumerate(fpn_features):
    print(f"FPN Level {i}: {feat.shape}")
```

## Training

### Image Classification

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import SwinTransformer
from utils import Config

# Configuration
config = Config.from_file('configs/swin_config.yaml')
model = SwinTransformer(config.to_dict())

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer and scheduler
optimizer = AdamW(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=0.05,
    betas=(0.9, 0.999)
)

# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=300, eta_min=1e-6
)

# Loss function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    scheduler.step()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch}, Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    
    # Save checkpoint
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'accuracy': accuracy
        }, f'swin_checkpoint_epoch_{epoch}.pth')
```

### Transfer Learning

```python
def create_swin_for_transfer_learning(num_classes, pretrained_path=None):
    """Create Swin Transformer for transfer learning."""
    
    # Load base configuration
    config = Config.from_file('configs/swin_config.yaml')
    config.set('num_classes', num_classes)
    
    # Create model
    model = SwinTransformer(config.to_dict())
    
    # Load pretrained weights if available
    if pretrained_path:
        checkpoint = torch.load(pretrained_path)
        
        # Remove classification head weights if number of classes differs
        state_dict = checkpoint['model_state_dict']
        if 'head.weight' in state_dict and state_dict['head.weight'].shape[0] != num_classes:
            del state_dict['head.weight']
            del state_dict['head.bias']
        
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    return model

# Fine-tuning with frozen backbone
def fine_tune_swin(model, train_dataloader, val_dataloader, freeze_backbone=True):
    """Fine-tune Swin Transformer with optional backbone freezing."""
    
    if freeze_backbone:
        # Freeze all parameters except classification head
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print("Backbone frozen, only training classification head")
    
    # Use smaller learning rate for fine-tuning
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch {epoch}, Train Loss: {train_loss/len(train_dataloader):.6f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')

# Usage
model = create_swin_for_transfer_learning(num_classes=10)
fine_tune_swin(model, train_dataloader, val_dataloader)
```

## Advanced Usage

### Window Attention Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_window_attention(model, image, layer_idx=0, block_idx=0, head_idx=0):
    """Visualize attention weights in Swin Transformer windows."""
    
    model.eval()
    
    # Add hooks to capture attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        if hasattr(module, 'attn') and hasattr(module.attn, 'attention_weights'):
            attention_weights.append(module.attn.attention_weights)
    
    # Register hook
    target_layer = model.layers[layer_idx].blocks[block_idx]
    hook = target_layer.register_forward_hook(attention_hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # Remove hook
    hook.remove()
    
    if attention_weights:
        # Get attention weights: [batch, num_windows, num_heads, window_size^2, window_size^2]
        attn = attention_weights[0][0, :, head_idx]  # First batch, specified head
        
        # Visualize attention for first few windows
        num_windows_to_show = min(4, attn.shape[0])
        
        fig, axes = plt.subplots(1, num_windows_to_show, figsize=(15, 3))
        for i in range(num_windows_to_show):
            window_attn = attn[i].cpu().numpy()
            
            if num_windows_to_show == 1:
                ax = axes
            else:
                ax = axes[i]
            
            im = ax.imshow(window_attn, cmap='Blues')
            ax.set_title(f'Window {i}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        plt.tight_layout()
        plt.show()

# Usage
config = Config.from_file('configs/swin_config.yaml')
model = SwinTransformer(config.to_dict())
image = torch.randn(3, 224, 224)

visualize_window_attention(model, image)
```

### Dynamic Window Size

```python
class AdaptiveSwinTransformer(SwinTransformer):
    """Swin Transformer with adaptive window size based on input resolution."""
    
    def __init__(self, config):
        super().__init__(config)
        self.base_window_size = config.get('window_size', 7)
    
    def forward(self, x):
        # Adapt window size based on input resolution
        _, _, H, W = x.shape
        base_size = self.img_size
        
        if H != base_size or W != base_size:
            # Scale window size proportionally
            scale_factor = min(H / base_size, W / base_size)
            new_window_size = max(1, int(self.base_window_size * scale_factor))
            
            # Update window size in all layers
            for layer in self.layers:
                for block in layer.blocks:
                    if hasattr(block.attn, 'window_size'):
                        block.attn.window_size = new_window_size
        
        return super().forward(x)

# Usage with different input sizes
adaptive_model = AdaptiveSwinTransformer(config.to_dict())

# Test with different resolutions
for size in [224, 384, 512]:
    test_input = torch.randn(1, 3, size, size)
    output = adaptive_model(test_input)
    print(f"Input size: {size}x{size}, Output shape: {output.shape}")
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, dataloader, num_epochs):
    """Train Swin Transformer with mixed precision for memory efficiency."""
    
    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

# Usage
train_with_mixed_precision(model, dataloader, num_epochs=100)
```

## Model Variants

### Official Model Configurations

```python
# Swin-T (Tiny)
swin_tiny = {
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 7
}

# Swin-S (Small)
swin_small = {
    'embed_dim': 96,
    'depths': [2, 2, 18, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 7
}

# Swin-B (Base)
swin_base = {
    'embed_dim': 128,
    'depths': [2, 2, 18, 2],
    'num_heads': [4, 8, 16, 32],
    'window_size': 7
}

# Swin-L (Large)
swin_large = {
    'embed_dim': 192,
    'depths': [2, 2, 18, 2],
    'num_heads': [6, 12, 24, 48],
    'window_size': 7
}
```

### Task-Specific Variants

```python
# For object detection (without classification head)
detection_config = swin_base.copy()
detection_config.update({
    'num_classes': 0,  # No classification head
    'global_pool': False
})

# For semantic segmentation
segmentation_config = swin_base.copy()
segmentation_config.update({
    'num_classes': 21,  # Number of segmentation classes
    'global_pool': False,
    'output_features': True
})

# For high-resolution inputs
high_res_config = swin_base.copy()
high_res_config.update({
    'img_size': 384,
    'window_size': 12  # Larger window for higher resolution
})
```

## Tips and Best Practices

### Training Tips

1. **Learning Rate**: Start with 1e-3 for training from scratch, 1e-4 for fine-tuning
2. **Warmup**: Use learning rate warmup for the first 20 epochs
3. **Weight Decay**: Use 0.05 weight decay for regularization
4. **Drop Path**: Gradually increase drop path rate (0.0 to 0.2)

### Data Augmentation

```python
import torchvision.transforms as transforms

# Recommended augmentation for Swin Transformer
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For validation/testing
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Performance Optimization

1. **Window Size**: Larger windows capture more global context but increase computation
2. **Patch Size**: Smaller patches increase resolution but quadratically increase computation
3. **Model Depth**: Deeper stages (stage 3) contribute most to performance

### Common Issues

1. **Memory Issues**: Reduce batch size, use gradient checkpointing, or mixed precision
2. **Slow Training**: Ensure efficient data loading, use multiple GPUs if available
3. **Poor Performance**: Check data augmentation, learning rate scheduling, and model configuration

## Paper Reference

- **Title**: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- **Authors**: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
- **Conference**: ICCV 2021
- **ArXiv**: https://arxiv.org/abs/2103.14030
- **Code**: https://github.com/microsoft/Swin-Transformer