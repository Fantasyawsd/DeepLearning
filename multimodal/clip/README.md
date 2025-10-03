# CLIP - Contrastive Language-Image Pre-training

## 简介

CLIP (Contrastive Language-Image Pre-training) 是OpenAI提出的多模态模型，通过对比学习将图像和文本映射到同一嵌入空间。

## 核心特性

- **对比学习**: 学习图像-文本对的匹配关系
- **零样本迁移**: 无需微调即可应用于新任务
- **大规模预训练**: 在4亿图像-文本对上训练
- **灵活应用**: 支持图像分类、检索、生成等多种任务

## 模型架构

```
图像编码器 (ViT/ResNet) → 图像嵌入 (512维)
文本编码器 (Transformer) → 文本嵌入 (512维)
→ 余弦相似度计算
→ 对比学习损失 (InfoNCE)
```

## 快速开始

### 安装依赖

```bash
pip install torch torchvision transformers
```

### 基本使用

```python
from multimodal.clip.model import CLIP
from utils import Config

# 加载配置
config = Config.from_file('multimodal/clip/config.yaml')

# 创建模型
model = CLIP(config.to_dict())

# 零样本图像分类
import torch
images = torch.randn(8, 3, 224, 224)
texts = ["a photo of a cat", "a photo of a dog"]

outputs = model(images, texts)
similarities = outputs['logits_per_image']
predictions = similarities.argmax(dim=-1)
```

## 应用示例

### 1. 零样本图像分类

```python
# 准备类别描述
class_names = ['cat', 'dog', 'bird', 'car', 'airplane']
text_prompts = [f"a photo of a {name}" for name in class_names]

# 编码
with torch.no_grad():
    text_features = model.encode_text(text_prompts)
    image_features = model.encode_image(images)

# 计算相似度
similarities = image_features @ text_features.t()
predictions = similarities.argmax(dim=-1)
```

### 2. 图像-文本检索

```python
# 文本到图像检索
query_text = "a beautiful sunset"
query_embedding = model.encode_text([query_text])

# 计算相似度
similarities = query_embedding @ image_embeddings.t()
top_k_images = similarities.topk(k=5).indices
```

### 3. 图像相似度搜索

```python
# 以图搜图
query_image = load_image('query.jpg')
query_embedding = model.encode_image(query_image)

# 在图像库中搜索
similarities = query_embedding @ database_embeddings.t()
similar_images = similarities.topk(k=10).indices
```

## 训练

### 准备数据

```python
from multimodal.clip.dataset import MultimodalDataset

dataset = MultimodalDataset(
    data_dir='path/to/data',
    transform=transform,
    tokenizer=tokenizer
)
```

### 训练脚本

```bash
python multimodal/clip/train.py \
    --config multimodal/clip/config.yaml \
    --output_dir outputs/clip \
    --batch_size 256 \
    --epochs 32
```

## 配置说明

```yaml
# config.yaml
model:
  image_encoder: 'vit_base_patch16_224'
  text_encoder: 'bert_base'
  embed_dim: 512
  
training:
  batch_size: 256
  learning_rate: 5e-4
  weight_decay: 0.2
  warmup_steps: 2000
  
data:
  image_size: 224
  max_text_length: 77
```

## 性能指标

| 数据集 | Zero-Shot Top-1 | Zero-Shot Top-5 |
|--------|----------------|----------------|
| ImageNet | 76.2% | 95.2% |
| CIFAR-10 | 94.9% | 99.8% |
| CIFAR-100 | 77.8% | 94.7% |

## 参考文献

```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}
```

## 许可证

MIT License
