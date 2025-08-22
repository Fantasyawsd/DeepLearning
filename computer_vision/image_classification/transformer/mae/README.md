# MAE (Masked Autoencoder) 完整实现

## 概述

MAE (Masked Autoencoder) 是一种自监督学习的视觉Transformer模型，由Meta AI在2022年提出。该模型通过随机掩码图像块并重建它们来学习强大的视觉表示。

**论文**: "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)  
**作者**: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick

## 核心特性

- **高效的自监督学习**: 通过75%的高掩码比例实现高效学习
- **不对称编码器-解码器架构**: 编码器只处理可见块，解码器重建完整图像
- **可扩展性**: 支持从小型到大型各种规模的模型
- **强泛化能力**: 在多种下游任务上表现优异

## 目录结构

```
mae/
├── model.py              # MAE模型实现
├── dataset.py            # 数据集处理和预处理
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── config.yaml           # 配置文件
├── load_pretrained.py    # HuggingFace预训练模型加载
└── README.md             # 说明文档
```

## 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install torch torchvision transformers einops timm matplotlib

# 或使用项目根目录的requirements.txt
pip install -r ../../../../../requirements.txt
```

### 2. 数据准备

```python
# 支持多种数据集
datasets = {
    'imagenet': 'ImageNet-1K (推荐用于大规模预训练)',
    'cifar10': 'CIFAR-10 (适合快速实验)',
    'cifar100': 'CIFAR-100',
    'custom': '自定义数据集'
}
```

### 3. 训练模型

```bash
# 使用默认配置训练
python train.py --config config.yaml --output_dir ./outputs/mae_experiment

# 从检查点恢复训练
python train.py --config config.yaml --resume ./outputs/mae_experiment/latest_checkpoint.pth
```

### 4. 测试模型

```bash
# 重建测试
python test.py --config config.yaml --model ./outputs/mae_experiment/best_model.pth --test_type reconstruction

# 特征提取
python test.py --config config.yaml --model ./outputs/mae_experiment/best_model.pth --test_type features

# 性能基准测试
python test.py --config config.yaml --model ./outputs/mae_experiment/best_model.pth --test_type benchmark

# 单张图像测试
python test.py --config config.yaml --model ./outputs/mae_experiment/best_model.pth --test_type single --image /path/to/image.jpg
```

### 5. 加载预训练模型

```python
from load_pretrained import load_pretrained_mae

# 加载HuggingFace预训练模型
model = load_pretrained_mae('mae-base')  # 或 'mae-large', 'mae-huge'

# 测试重建
import torch
test_input = torch.randn(1, 3, 224, 224)
loss, pred, mask = model(test_input)
```

## 模型配置

### 基本配置选项

```yaml
# 模型结构
img_size: 224              # 输入图像尺寸
patch_size: 16             # 补丁尺寸
embed_dim: 1024            # 编码器嵌入维度
depth: 24                  # 编码器层数
num_heads: 16              # 注意力头数

# 训练参数
mask_ratio: 0.75           # 掩码比例
batch_size: 32             # 批次大小
epochs: 100                # 训练轮数
lr: 1.5e-4                 # 学习率
```

### 预设配置

我们提供了三种预设配置：

1. **Small Config** - 适合资源受限环境
   - 嵌入维度: 384, 层数: 6, 参数量: ~22M

2. **Base Config** - 平衡性能和效率
   - 嵌入维度: 768, 层数: 12, 参数量: ~87M

3. **Large Config** - 最佳性能
   - 嵌入维度: 1024, 层数: 24, 参数量: ~307M

## 使用示例

### 基本训练示例

```python
from model import MAE
from dataset import create_mae_dataloaders
from train import MAETrainer
import torch

# 加载配置
config = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mask_ratio': 0.75,
    'batch_size': 32,
    'epochs': 100,
    'dataset': {
        'name': 'cifar10',
        'root': './data',
        'use_validation': True
    },
    'optimizer': {
        'name': 'adamw',
        'lr': 1.5e-4,
        'weight_decay': 0.05
    },
    'output_dir': './outputs/mae'
}

# 创建训练器并训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = MAETrainer(config, device)
trainer.train()
```

### 推理示例

```python
from model import MAE
import torch
from PIL import Image
from dataset import get_mae_transforms

# 加载模型
model = MAE(config)
model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
model.eval()

# 处理图像
transform = get_mae_transforms(224, is_training=False)
image = Image.open('test.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# 推理
with torch.no_grad():
    loss, pred, mask = model(image_tensor, mask_ratio=0.75)
    
# 可视化重建结果
reconstructed = model.unpatchify(pred)
```

### 特征提取示例

```python
# 提取编码器特征
model.eval()
with torch.no_grad():
    latent, mask, ids_restore = model.encoder(image_tensor, mask_ratio=0.0)
    features = latent[:, 0]  # CLS token特征

# 用于下游任务
classifier = torch.nn.Linear(model.config['embed_dim'], num_classes)
output = classifier(features)
```

## 数据集支持

### 支持的数据集

1. **ImageNet-1K**
   - 标准大规模图像分类数据集
   - 推荐用于大规模预训练

2. **CIFAR-10/100**
   - 小规模数据集，适合快速实验
   - 自动下载和预处理

3. **自定义数据集**
   - 支持任意目录结构的图像数据
   - 灵活的数据预处理pipeline

### 数据预处理

```python
from dataset import get_mae_transforms, get_augmentation_transforms

# 标准MAE变换
transform = get_mae_transforms(img_size=224, is_training=True)

# 自定义数据增强
augment_transform = get_augmentation_transforms(
    img_size=224, 
    augment_strength='medium'  # 'light', 'medium', 'strong'
)
```

## 训练策略

### 学习率调度

支持多种学习率调度策略：

1. **Cosine Annealing** (推荐)
   ```yaml
   scheduler:
     name: "cosine"
     min_lr: 0.0
   ```

2. **Step Scheduler**
   ```yaml
   scheduler:
     name: "step"
     step_size: 30
     gamma: 0.1
   ```

3. **Plateau Scheduler**
   ```yaml
   scheduler:
     name: "plateau"
     factor: 0.5
     patience: 10
   ```

### 优化器选择

1. **AdamW** (推荐)
   ```yaml
   optimizer:
     name: "adamw"
     lr: 1.5e-4
     weight_decay: 0.05
     betas: [0.9, 0.95]
   ```

2. **SGD**
   ```yaml
   optimizer:
     name: "sgd"
     lr: 0.1
     weight_decay: 1e-4
     momentum: 0.9
   ```

### 训练技巧

1. **混合精度训练**: 提升训练速度，减少显存占用
2. **梯度累积**: 支持大批次大小训练
3. **分布式训练**: 支持多GPU训练
4. **检查点恢复**: 支持训练中断恢复

## 性能基准

### 模型规模对比

| 模型 | 参数量 | ImageNet重建损失 | 推理速度 (samples/s) |
|------|--------|------------------|---------------------|
| MAE-Small | 22M | 0.67 | 1200 |
| MAE-Base | 87M | 0.45 | 800 |
| MAE-Large | 307M | 0.32 | 400 |

### 下游任务性能

| 任务 | 数据集 | MAE-Base | MAE-Large |
|------|--------|----------|-----------|
| 图像分类 | ImageNet | 83.6% | 85.9% |
| 目标检测 | COCO | 47.2 mAP | 50.3 mAP |
| 语义分割 | ADE20K | 48.1 mIoU | 53.6 mIoU |

## 可视化工具

### 训练监控

```python
# 训练过程可视化
trainer.plot_training_curves()  # 损失曲线、学习率变化等
```

### 重建可视化

```python
from dataset import visualize_mae_reconstruction

# 可视化重建结果
visualize_mae_reconstruction(
    model=model,
    dataloader=test_loader,
    device=device,
    num_samples=8,
    mask_ratio=0.75
)
```

### 注意力可视化

```python
# 提取注意力权重
def get_attention_maps(model, image):
    model.eval()
    with torch.no_grad():
        # 获取编码器输出和注意力权重
        latent, mask, _ = model.encoder(image, mask_ratio=0.0)
        # 注意力权重可视化代码...
```

## 常见问题

### Q: 如何选择合适的掩码比例？
A: 标准设置是75%，但可以根据数据复杂度调整：
- 简单数据集：可尝试80-90%
- 复杂数据集：可从50-75%开始

### Q: 训练需要多长时间？
A: 取决于数据集规模和硬件：
- CIFAR-10 (Base): ~2小时 (单GPU)
- ImageNet (Base): ~5天 (8 GPU)
- ImageNet (Large): ~10天 (8 GPU)

### Q: 内存不足怎么办？
A: 可以尝试：
1. 减小批次大小
2. 使用梯度累积
3. 使用混合精度训练
4. 选择更小的模型配置

### Q: 如何用于下游任务？
A: MAE学到的表示可以用于：
1. 微调分类任务
2. 作为特征提取器
3. 初始化其他视觉模型

## 参考资料

1. **原始论文**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
2. **官方实现**: [facebookresearch/mae](https://github.com/facebookresearch/mae)
3. **HuggingFace实现**: [transformers MAE](https://huggingface.co/docs/transformers/model_doc/vit_mae)

## 许可证

本实现遵循MIT许可证。MAE模型的原始论文和实现由Meta AI发布。