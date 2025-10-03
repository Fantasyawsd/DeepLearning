# BLIP - Bootstrapping Language-Image Pre-training

## 简介

BLIP (Bootstrapping Language-Image Pre-training)是一个统一的视觉-语言预训练框架，支持理解和生成任务。

## 核心特性

- **统一架构**: 同时支持理解和生成任务
- **CapFilt**: 自动生成和过滤图像描述
- **多任务学习**: ITC、ITM、LM三种预训练目标
- **灵活应用**: 图像描述、VQA、检索等

## 模型架构

```
视觉编码器 (ViT) + 文本编码器 (BERT) + 文本解码器
↓
三种预训练任务：
1. ITC (Image-Text Contrastive)
2. ITM (Image-Text Matching)
3. LM (Language Modeling)
```

## 快速开始

### 图像描述生成

```python
from multimodal.blip.model import BLIP

# 加载模型
blip = BLIP.from_pretrained('blip-base')

# 生成描述
import torch
from PIL import Image

image = Image.open('example.jpg')
caption = blip.generate_caption(image)
print(f"描述: {caption}")
```

### 视觉问答 (VQA)

```python
# 准备图像和问题
image = Image.open('scene.jpg')
question = "What is the person doing in the image?"

# 生成答案
answer = blip.answer_question(image, question)
print(f"答案: {answer}")
```

### 图像-文本检索

```python
# 文本到图像检索
query_text = "a dog playing in the park"
top_images = blip.retrieve_images(query_text, image_database, top_k=5)

# 图像到文本检索
query_image = Image.open('query.jpg')
top_texts = blip.retrieve_texts(query_image, text_database, top_k=5)
```

## 应用场景

### 1. 自动图像标注

```python
class ImageCaptioner:
    def __init__(self):
        self.blip = BLIP.from_pretrained('blip-base')
    
    def caption_batch(self, images, num_beams=3):
        captions = []
        for image in images:
            caption = self.blip.generate_caption(
                image,
                num_beams=num_beams,
                max_length=20
            )
            captions.append(caption)
        return captions

# 使用
captioner = ImageCaptioner()
images = load_image_batch('dataset/')
captions = captioner.caption_batch(images)
```

### 2. 智能搜索引擎

```python
class MultimodalSearch:
    def __init__(self):
        self.blip = BLIP.from_pretrained('blip-base')
        self.image_features = None
        self.text_features = None
    
    def index_images(self, image_paths):
        # 提取图像特征
        features = []
        for path in image_paths:
            image = Image.open(path)
            feat = self.blip.encode_image(image)
            features.append(feat)
        self.image_features = torch.stack(features)
    
    def search(self, query, top_k=10):
        # 支持文本或图像查询
        if isinstance(query, str):
            query_feat = self.blip.encode_text(query)
        else:
            query_feat = self.blip.encode_image(query)
        
        # 计算相似度
        similarities = query_feat @ self.image_features.t()
        top_indices = similarities.topk(k=top_k).indices
        return top_indices
```

### 3. 对话式图像理解

```python
class VisualAssistant:
    def __init__(self):
        self.blip = BLIP.from_pretrained('blip-vqa')
        self.conversation_history = []
    
    def understand_image(self, image):
        # 生成初始描述
        description = self.blip.generate_caption(image)
        self.conversation_history.append({
            'type': 'description',
            'content': description
        })
        return description
    
    def answer_question(self, image, question):
        # 回答关于图像的问题
        answer = self.blip.answer_question(image, question)
        self.conversation_history.append({
            'type': 'qa',
            'question': question,
            'answer': answer
        })
        return answer

# 使用示例
assistant = VisualAssistant()
image = Image.open('vacation_photo.jpg')

# 初始理解
desc = assistant.understand_image(image)
print(f"图像描述: {desc}")

# 多轮问答
questions = [
    "Where was this photo taken?",
    "What is the weather like?",
    "How many people are in the photo?"
]

for q in questions:
    answer = assistant.answer_question(image, q)
    print(f"Q: {q}\nA: {answer}\n")
```

## 训练

### 多任务训练

```python
# 训练配置
config = {
    'loss_weights': {
        'itc': 1.0,  # 对比学习
        'itm': 1.0,  # 匹配任务
        'lm': 1.0    # 语言建模
    }
}

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch['images']
        texts = batch['texts']
        
        # ITC损失
        loss_itc = model.compute_itc_loss(images, texts)
        
        # ITM损失
        loss_itm = model.compute_itm_loss(images, texts)
        
        # LM损失
        loss_lm = model.compute_lm_loss(images, texts)
        
        # 总损失
        loss = (config['loss_weights']['itc'] * loss_itc +
                config['loss_weights']['itm'] * loss_itm +
                config['loss_weights']['lm'] * loss_lm)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### CapFilt数据增强

```python
# 使用模型生成和过滤描述
def capfilt_augmentation(images, model):
    # 生成描述
    synthetic_captions = []
    for image in images:
        caption = model.generate_caption(image)
        synthetic_captions.append(caption)
    
    # 过滤低质量描述
    filtered_pairs = []
    for image, caption in zip(images, synthetic_captions):
        # 计算图像-文本匹配分数
        score = model.compute_itm_score(image, caption)
        if score > threshold:
            filtered_pairs.append((image, caption))
    
    return filtered_pairs
```

## 性能指标

### 图像描述任务

| 数据集 | BLEU-4 | METEOR | CIDEr | SPICE |
|--------|--------|--------|-------|-------|
| COCO | 38.6 | 29.4 | 129.7 | 23.2 |
| Flickr30k | 32.4 | 26.1 | 98.3 | 21.6 |

### VQA任务

| 数据集 | Accuracy |
|--------|----------|
| VQAv2 | 77.5% |
| GQA | 61.2% |

### 检索任务

| 数据集 | Image→Text R@1 | Text→Image R@1 |
|--------|----------------|----------------|
| COCO | 82.4 | 65.1 |
| Flickr30k | 95.3 | 84.8 |

## 参考文献

```bibtex
@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  booktitle={ICML},
  year={2022}
}
```

## 许可证

MIT License
