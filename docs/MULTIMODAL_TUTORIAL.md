# 多模态大模型完整教程：从入门到精通

## 📚 教程概览

本教程将带您从基础到高级，系统学习多模态大模型的理论、实现和应用。涵盖计算机视觉、自然语言处理以及多模态融合的经典网络、模型和技术。

### 🎯 学习目标

- 掌握深度学习基础和经典网络架构
- 理解计算机视觉和自然语言处理的核心技术
- 学习多模态模型的原理和实现
- 掌握从零开始构建和训练多模态模型
- 了解最新的多模态大模型技术

### 📖 教程结构

1. **基础篇** - 深度学习基础和经典网络
2. **视觉篇** - 计算机视觉模型和技术
3. **语言篇** - 自然语言处理模型
4. **多模态篇** - 多模态模型和应用
5. **进阶篇** - 高级技术和最新研究

---

## 第一章：深度学习基础篇

### 1.1 神经网络基础

#### 1.1.1 感知机与多层感知机

**核心概念：**
- 神经元模型：输入、权重、偏置、激活函数
- 前向传播：从输入到输出的计算过程
- 反向传播：梯度下降和权重更新

**示例代码：**
```python
import torch
import torch.nn as nn

# 简单的多层感知机
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleMLP()
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

#### 1.1.2 常用激活函数

**激活函数对比：**
- **ReLU**: `f(x) = max(0, x)` - 最常用，解决梯度消失
- **GELU**: `f(x) = x * Φ(x)` - Transformer中常用
- **Sigmoid**: `f(x) = 1/(1+e^(-x))` - 二分类输出
- **Tanh**: `f(x) = (e^x - e^(-x))/(e^x + e^(-x))` - 输出范围[-1,1]

**代码示例：**
```python
import torch.nn.functional as F

x = torch.randn(10)
print(f"ReLU: {F.relu(x)}")
print(f"GELU: {F.gelu(x)}")
print(f"Sigmoid: {F.sigmoid(x)}")
print(f"Tanh: {F.tanh(x)}")
```

#### 1.1.3 损失函数

**常用损失函数：**
- **交叉熵损失** (分类任务)
- **均方误差损失** (回归任务)
- **对比学习损失** (多模态学习)

```python
# 分类任务
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, labels)

# 回归任务
criterion = nn.MSELoss()
loss = criterion(predictions, targets)
```

### 1.2 经典卷积神经网络 (CNN)

#### 1.2.1 LeNet-5 (1998) - CNN的先驱

**模型特点：**
- 首个成功的卷积神经网络
- 用于手写数字识别 (MNIST)
- 包含卷积层、池化层和全连接层

**网络结构：**
```
输入(28x28) → Conv1(5x5, 6) → Pool(2x2) → Conv2(5x5, 16) → Pool(2x2) → FC(120) → FC(84) → FC(10)
```

**实现示例：**
```python
from models import LeNet5
from utils import Config

# 创建LeNet模型
config = Config.from_file('configs/lenet_config.yaml')
model = LeNet5(config.to_dict())

# 模型信息
print(f"LeNet-5 参数量: {model.count_parameters():,}")
model.summary()

# 训练示例
import torch
images = torch.randn(32, 1, 28, 28)  # MNIST格式
outputs = model(images)
print(f"输出形状: {outputs['logits'].shape}")  # [32, 10]
```

**实践任务：**
1. 在MNIST数据集上训练LeNet-5
2. 可视化卷积层学到的特征
3. 尝试改进网络结构（添加BatchNorm、Dropout）

#### 1.2.2 ResNet (2015) - 残差网络

**核心创新：**
- **残差连接 (Skip Connection)**: 解决深度网络梯度消失问题
- **批归一化 (Batch Normalization)**: 加速训练
- **瓶颈结构 (Bottleneck)**: 减少参数量

**残差块原理：**
```
y = F(x) + x  # F(x)是残差映射，x是恒等映射
```

**网络架构：**
- ResNet-18/34: 使用BasicBlock (两层卷积)
- ResNet-50/101/152: 使用Bottleneck (三层卷积，1x1降维→3x3卷积→1x1升维)

**使用示例：**
```python
from models import ResNet
from computer_vision.image_classification.cnn.resnet.model import create_resnet

# 创建不同深度的ResNet
resnet18 = create_resnet('resnet18', num_classes=1000)
resnet50 = create_resnet('resnet50', num_classes=1000)
resnet152 = create_resnet('resnet152', num_classes=1000)

print(f"ResNet-18 参数量: {sum(p.numel() for p in resnet18.parameters()):,}")
print(f"ResNet-50 参数量: {sum(p.numel() for p in resnet50.parameters()):,}")
print(f"ResNet-152 参数量: {sum(p.numel() for p in resnet152.parameters()):,}")

# 前向传播
images = torch.randn(16, 3, 224, 224)
output = resnet50(images)
print(f"输出形状: {output['logits'].shape}")  # [16, 1000]
```

**实践任务：**
1. 在CIFAR-10数据集上训练ResNet-18
2. 对比不同深度ResNet的性能
3. 可视化残差块的特征图

---

## 第二章：计算机视觉篇

### 2.1 Vision Transformer (ViT) - 视觉领域的革命

#### 2.1.1 从CNN到Transformer

**Transformer在视觉中的应用：**
- 将图像分割成patches (通常16x16)
- 每个patch展平并线性投影得到token
- 添加位置编码
- 通过多层Transformer编码器处理

**ViT架构：**
```
图像(224x224x3) → Patch Embedding(14x14 patches) → Position Encoding 
→ Transformer Encoder × 12 → MLP Head → 分类结果
```

**代码实现：**
```python
from computer_vision.image_classification.transformer.vit.model import create_vit

# 创建ViT模型
vit_base = create_vit('vit_base_patch16_224', {
    'num_classes': 1000,
    'image_size': 224,
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12
})

# 模型信息
print(f"ViT-Base 参数量: {sum(p.numel() for p in vit_base.parameters()):,}")

# 前向传播
images = torch.randn(8, 3, 224, 224)
output = vit_base(images)
print(f"输出形状: {output.shape}")  # [8, 1000]
```

#### 2.1.2 注意力机制可视化

```python
# 获取注意力权重
from computer_vision.image_classification.transformer.vit.model import VisionTransformer

model = VisionTransformer(config)
images = torch.randn(1, 3, 224, 224)

# 前向传播并获取注意力
with torch.no_grad():
    output, attentions = model(images, output_attentions=True)

# 可视化第一层的注意力
import matplotlib.pyplot as plt
attention_map = attentions[0][0, 0].cpu()  # 第一个样本，第一个头
plt.imshow(attention_map, cmap='viridis')
plt.title('ViT Attention Map - Layer 1, Head 1')
plt.colorbar()
plt.savefig('vit_attention.png')
```

### 2.2 MAE (Masked Autoencoder) - 自监督学习

#### 2.2.1 MAE原理

**核心思想：**
- 随机掩码图像的75%
- 编码器只处理可见的patches
- 解码器重建被掩码的patches
- 学习强大的视觉表示

**网络结构：**
```
输入图像 → Patch分割 → 随机掩码(75%) 
→ 编码器(ViT) → 解码器(轻量级Transformer) → 重建图像
```

**实现示例：**
```python
from computer_vision.image_classification.transformer.mae.model import MAE
from computer_vision.image_classification.transformer.mae.load_pretrained import load_pretrained_mae

# 创建MAE模型
mae = MAE({
    'image_size': 224,
    'patch_size': 16,
    'embed_dim': 768,
    'encoder_depth': 12,
    'decoder_depth': 8,
    'mask_ratio': 0.75
})

# 或加载预训练模型
mae_pretrained = load_pretrained_mae('mae-base')

# 训练（自监督）
images = torch.randn(16, 3, 224, 224)
outputs = mae(images)
loss = outputs['loss']
reconstructed = outputs['pred']
mask = outputs['mask']

print(f"重建损失: {loss.item():.4f}")
print(f"重建图像形状: {reconstructed.shape}")
```

#### 2.2.2 MAE预训练和微调

**预训练阶段：**
```python
# 自监督预训练
for epoch in range(num_epochs):
    for images in dataloader:
        outputs = mae(images)
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**微调阶段：**
```python
# 在下游任务上微调
mae_pretrained = load_pretrained_mae('mae-base')
encoder = mae_pretrained.encoder  # 只使用编码器

# 添加分类头
classifier = nn.Sequential(
    nn.LayerNorm(768),
    nn.Linear(768, num_classes)
)

# 微调训练
for images, labels in dataloader:
    features = encoder(images)
    logits = classifier(features)
    loss = criterion(logits, labels)
    # ... 优化步骤
```

### 2.3 目标检测 - YOLO系列

#### 2.3.1 YOLOv1 - 实时目标检测

**YOLO核心思想：**
- 将检测问题转化为回归问题
- 将图像划分为SxS网格
- 每个网格预测B个边界框和类别概率
- 端到端训练，实时推理

**网络输出：**
```
输出张量: [batch, S, S, B*(5+C)]
- 5 = (x, y, w, h, confidence)
- C = 类别数量
```

**使用示例：**
```python
from computer_vision.object_detection.yolo_series.yolov1.model import YOLOv1

# 创建YOLO模型
yolo = YOLOv1({
    'num_classes': 20,  # PASCAL VOC
    'S': 7,  # 网格大小
    'B': 2,  # 每个网格的边界框数
    'C': 20  # 类别数
})

# 检测
images = torch.randn(8, 3, 448, 448)
predictions = yolo(images)
print(f"预测形状: {predictions.shape}")  # [8, 7, 7, 30]

# 后处理：非极大值抑制 (NMS)
from computer_vision.object_detection.yolo_series.yolov1.utils import nms
boxes, scores, labels = nms(predictions, conf_threshold=0.5, nms_threshold=0.4)
```

---

## 第三章：自然语言处理篇

### 3.1 Transformer基础

#### 3.1.1 自注意力机制 (Self-Attention)

**核心公式：**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**多头注意力 (Multi-Head Attention)：**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        # Q, K, V投影
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
```

#### 3.1.2 位置编码

**正弦位置编码：**
```python
def get_sinusoid_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                        (-math.log(10000.0) / d_model))
    
    pos_encoding = torch.zeros(seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding
```

**可学习位置编码：**
```python
self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
```

### 3.2 GPT - 生成式预训练Transformer

#### 3.2.1 GPT架构

**核心特点：**
- 自回归语言模型
- 单向注意力（因果掩码）
- 预训练 + 微调范式

**网络结构：**
```
Token Embedding + Position Embedding
→ Transformer Decoder × N
→ Language Model Head
```

**使用示例：**
```python
from nlp.language_models.gpt_series.gpt.model import create_gpt

# 创建GPT模型
gpt = create_gpt('gpt-small', {
    'vocab_size': 50257,
    'n_positions': 1024,
    'n_embd': 768,
    'n_layer': 12,
    'n_head': 12
})

# 文本生成
input_ids = torch.randint(0, 50257, (1, 10))  # 输入token
generated = gpt.generate(
    input_ids,
    max_length=50,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)

print(f"生成的token: {generated}")
```

#### 3.2.2 文本生成策略

**贪婪解码：**
```python
def greedy_decode(model, input_ids, max_length):
    for _ in range(max_length):
        logits = model(input_ids)['logits']
        next_token = logits[:, -1, :].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
    return input_ids
```

**Top-k采样：**
```python
def top_k_sampling(logits, k=50):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, 1)
    return top_k_indices.gather(-1, next_token_idx)
```

**Top-p (Nucleus) 采样：**
```python
def top_p_sampling(logits, p=0.95):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 移除累积概率超过p的token
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 采样
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)
```

---

## 第四章：多模态模型篇

### 4.1 CLIP - 连接视觉和语言

#### 4.1.1 CLIP原理

**核心思想：**
- 对比学习：图像和文本在共享嵌入空间中对齐
- 大规模预训练：4亿图像-文本对
- 零样本迁移：无需微调即可应用新任务

**网络架构：**
```
图像编码器 (ViT/ResNet) → 图像嵌入 (d维)
文本编码器 (Transformer) → 文本嵌入 (d维)
→ 对比学习损失 (InfoNCE)
```

**CLIP实现框架：**
```python
# multimodal/clip/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = self._build_image_encoder(config)
        self.text_encoder = self._build_text_encoder(config)
        
        # 投影层
        self.image_projection = nn.Linear(config['image_dim'], config['embed_dim'])
        self.text_projection = nn.Linear(config['text_dim'], config['embed_dim'])
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_image(self, images):
        image_features = self.image_encoder(images)
        image_embeds = self.image_projection(image_features)
        return F.normalize(image_embeds, dim=-1)
    
    def encode_text(self, text_tokens):
        text_features = self.text_encoder(text_tokens)
        text_embeds = self.text_projection(text_features)
        return F.normalize(text_embeds, dim=-1)
    
    def forward(self, images, text_tokens):
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(text_tokens)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        return {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'image_embeds': image_embeds,
            'text_embeds': text_embeds
        }
```

#### 4.1.2 对比学习损失

**InfoNCE损失：**
```python
def contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    loss = (loss_i + loss_t) / 2
    return loss
```

#### 4.1.3 CLIP应用示例

**零样本图像分类：**
```python
from multimodal.clip.model import CLIP

# 加载预训练CLIP
clip_model = CLIP(config)
clip_model.load_checkpoint('clip_pretrained.pth')

# 准备类别文本
class_names = ['cat', 'dog', 'bird', 'car', 'airplane']
text_prompts = [f"a photo of a {name}" for name in class_names]
text_tokens = tokenizer(text_prompts)

# 编码文本
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)

# 编码图像
image = load_image('test.jpg')
image_features = clip_model.encode_image(image)

# 计算相似度
similarities = (image_features @ text_features.t()).softmax(dim=-1)
predicted_class = class_names[similarities.argmax()]

print(f"预测类别: {predicted_class}")
print(f"置信度: {similarities.max():.2%}")
```

**图像-文本检索：**
```python
# 文本到图像检索
query_text = "a beautiful sunset over the ocean"
query_tokens = tokenizer([query_text])
query_embedding = clip_model.encode_text(query_tokens)

# 计算与所有图像的相似度
image_embeddings = clip_model.encode_image(all_images)
similarities = query_embedding @ image_embeddings.t()
top_k_indices = similarities.topk(k=5).indices

print(f"最相似的图像: {top_k_indices}")
```

### 4.2 DALL-E - 文本到图像生成

#### 4.2.1 DALL-E架构

**核心组件：**
1. **dVAE (discrete VAE)**: 将图像编码为离散token
2. **Transformer**: 自回归生成图像token
3. **文本编码器**: 编码条件文本

**生成流程：**
```
文本 → Text Encoder → 文本token
→ Transformer (自回归) → 图像token
→ dVAE Decoder → 生成图像
```

**实现框架：**
```python
# multimodal/dalle/model.py
class DALLE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(config)
        self.image_tokenizer = dVAE(config)
        self.transformer = GPT(config)
        
    def forward(self, text, images=None):
        # 编码文本
        text_tokens = self.text_encoder(text)
        
        if images is not None:
            # 训练模式：编码图像为离散token
            image_tokens = self.image_tokenizer.encode(images)
            
            # 拼接文本和图像token
            tokens = torch.cat([text_tokens, image_tokens], dim=1)
            
            # 自回归预测
            logits = self.transformer(tokens)
            
            # 计算损失
            loss = F.cross_entropy(
                logits[:, text_tokens.shape[1]-1:-1].reshape(-1, logits.shape[-1]),
                image_tokens.reshape(-1)
            )
            return {'loss': loss}
        else:
            # 生成模式
            generated_tokens = self.generate(text_tokens)
            generated_images = self.image_tokenizer.decode(generated_tokens)
            return {'images': generated_images}
    
    @torch.no_grad()
    def generate(self, text_tokens, num_images=1):
        # 自回归生成图像token
        batch_size = text_tokens.shape[0]
        generated = text_tokens
        
        for _ in range(self.config['image_token_length']):
            logits = self.transformer(generated)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        
        # 解码为图像
        image_tokens = generated[:, text_tokens.shape[1]:]
        images = self.image_tokenizer.decode(image_tokens)
        return images
```

#### 4.2.2 使用示例

```python
from multimodal.dalle.model import DALLE

# 创建DALL-E模型
dalle = DALLE(config)

# 文本到图像生成
text_prompt = "a painting of a cat wearing a hat in the style of Van Gogh"
text_tokens = tokenizer([text_prompt])

# 生成图像
generated_images = dalle.generate(text_tokens, num_images=4)

# 保存结果
for i, img in enumerate(generated_images):
    save_image(img, f'dalle_output_{i}.png')
```

### 4.3 BLIP - 统一的视觉-语言理解和生成

#### 4.3.1 BLIP架构

**核心创新：**
- **多任务预训练**: 理解、生成、检索三合一
- **CapFilt**: 自动生成和过滤图像描述
- **统一架构**: 共享的图像编码器

**三种模式：**
1. **图像-文本对比学习** (ITC)
2. **图像-文本匹配** (ITM)
3. **图像条件的语言建模** (LM)

**实现框架：**
```python
# multimodal/blip/model.py
class BLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_encoder = ViT(config)
        self.text_encoder = BERTEncoder(config)
        self.text_decoder = BERTDecoder(config)
        
        # 多任务头
        self.itm_head = nn.Linear(config['hidden_size'], 2)  # 匹配/不匹配
        self.itc_proj = nn.Linear(config['hidden_size'], config['embed_dim'])
    
    def forward(self, images, text_tokens, mode='itc'):
        # 编码图像
        image_embeds = self.visual_encoder(images)
        
        if mode == 'itc':
            # 图像-文本对比
            return self.compute_itc(image_embeds, text_tokens)
        elif mode == 'itm':
            # 图像-文本匹配
            return self.compute_itm(image_embeds, text_tokens)
        elif mode == 'lm':
            # 图像描述生成
            return self.generate_caption(image_embeds, text_tokens)
    
    def compute_itc(self, image_embeds, text_tokens):
        # 对比学习
        image_feat = F.normalize(self.itc_proj(image_embeds[:, 0]), dim=-1)
        text_embeds = self.text_encoder(text_tokens)
        text_feat = F.normalize(self.itc_proj(text_embeds[:, 0]), dim=-1)
        
        sim = image_feat @ text_feat.t()
        return {'similarity': sim}
    
    def generate_caption(self, image_embeds, max_length=20):
        # 自回归生成描述
        batch_size = image_embeds.shape[0]
        input_ids = torch.ones((batch_size, 1), dtype=torch.long) * self.bos_token_id
        
        for _ in range(max_length):
            outputs = self.text_decoder(input_ids, encoder_hidden_states=image_embeds)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            
            if (next_token == self.eos_token_id).all():
                break
        
        return input_ids
```

#### 4.3.2 应用示例

**图像描述生成：**
```python
from multimodal.blip.model import BLIP

# 加载BLIP模型
blip = BLIP(config)
blip.load_checkpoint('blip_pretrained.pth')

# 生成图像描述
image = load_image('example.jpg')
caption_ids = blip.generate_caption(image)
caption = tokenizer.decode(caption_ids[0])

print(f"生成的描述: {caption}")
```

**视觉问答 (VQA)：**
```python
# 准备问题
question = "What color is the cat in the image?"
question_tokens = tokenizer([question])

# 编码图像和问题
image_embeds = blip.visual_encoder(image)
question_embeds = blip.text_encoder(question_tokens)

# 生成答案
answer = blip.generate_answer(image_embeds, question_embeds)
print(f"答案: {answer}")
```

---

## 第五章：进阶技术篇

### 5.1 高效微调技术

#### 5.1.1 LoRA (Low-Rank Adaptation)

**核心思想：**
- 冻结预训练权重
- 添加低秩矩阵进行微调
- 大幅减少可训练参数

**实现：**
```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 1.0 / rank
        
    def forward(self, x):
        # 原始线性层（冻结） + LoRA
        return x @ (self.lora_A @ self.lora_B) * self.scaling

# 应用到模型
def apply_lora(model, rank=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 替换为LoRA版本
            lora_layer = LoRALayer(module.in_features, module.out_features, rank)
            # ... 替换逻辑
```

#### 5.1.2 Adapter Tuning

**Adapter结构：**
```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # 残差连接
```

### 5.2 零样本和少样本学习

#### 5.2.1 Prompt Engineering

**视觉提示：**
```python
# CLIP零样本分类的提示工程
templates = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "a photo of a clean {}",
    "a dark photo of the {}",
]

def get_text_features(class_names, model):
    text_features = []
    for class_name in class_names:
        texts = [template.format(class_name) for template in templates]
        text_tokens = tokenizer(texts)
        class_features = model.encode_text(text_tokens)
        class_features = class_features.mean(dim=0)  # 平均多个提示
        text_features.append(class_features)
    
    return torch.stack(text_features)
```

#### 5.2.2 Few-Shot Learning

**元学习示例：**
```python
def few_shot_learning(model, support_set, query_set, k_shot=5):
    # 支持集：每类k个样本
    support_images, support_labels = support_set
    query_images, query_labels = query_set
    
    # 提取特征
    support_features = model.encode_image(support_images)
    query_features = model.encode_image(query_images)
    
    # 计算原型（每类特征的均值）
    prototypes = []
    for class_id in support_labels.unique():
        class_mask = support_labels == class_id
        class_features = support_features[class_mask]
        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)
    
    # 最近邻分类
    similarities = query_features @ prototypes.t()
    predictions = similarities.argmax(dim=-1)
    
    accuracy = (predictions == query_labels).float().mean()
    return accuracy
```

### 5.3 多模态数据增强

#### 5.3.1 图像增强

```python
import torchvision.transforms as T

# CLIP训练的增强策略
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### 5.3.2 文本增强

```python
# 同义词替换
def synonym_replacement(text, n=3):
    words = text.split()
    for _ in range(n):
        idx = random.randint(0, len(words) - 1)
        synonyms = get_synonyms(words[idx])
        if synonyms:
            words[idx] = random.choice(synonyms)
    return ' '.join(words)

# 回译增强
def back_translation(text, intermediate_lang='de'):
    # 英语 → 德语 → 英语
    translated = translate(text, target=intermediate_lang)
    back_translated = translate(translated, target='en')
    return back_translated
```

### 5.4 模型评估指标

#### 5.4.1 图像-文本检索指标

```python
def compute_retrieval_metrics(image_embeds, text_embeds, k=[1, 5, 10]):
    # 计算相似度矩阵
    similarity = image_embeds @ text_embeds.t()
    
    # 图像到文本检索
    i2t_ranks = []
    for i in range(len(image_embeds)):
        ranks = similarity[i].argsort(descending=True)
        i2t_ranks.append((ranks == i).nonzero()[0].item())
    
    # 计算Recall@K
    recalls = {}
    for k_val in k:
        recalls[f'R@{k_val}'] = (torch.tensor(i2t_ranks) < k_val).float().mean()
    
    return recalls
```

#### 5.4.2 图像生成指标

```python
# FID (Fréchet Inception Distance)
def calculate_fid(real_images, generated_images, inception_model):
    # 提取Inception特征
    real_features = inception_model(real_images)
    gen_features = inception_model(generated_images)
    
    # 计算均值和协方差
    mu_real = real_features.mean(dim=0)
    mu_gen = gen_features.mean(dim=0)
    sigma_real = torch.cov(real_features.t())
    sigma_gen = torch.cov(gen_features.t())
    
    # FID计算
    diff = mu_real - mu_gen
    covmean = torch.linalg.matrix_power(sigma_real @ sigma_gen, 0.5)
    
    fid = diff @ diff + torch.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid.item()
```

---

## 第六章：实战项目

### 6.1 项目1：构建图像搜索引擎

**目标：** 使用CLIP构建基于文本的图像搜索系统

**步骤：**
```python
# 1. 预处理图像库
from multimodal.clip.model import CLIP

clip_model = CLIP.from_pretrained('clip-vit-base')

image_database = []
image_embeddings = []

for image_path in image_library:
    image = load_and_preprocess(image_path)
    embedding = clip_model.encode_image(image)
    
    image_database.append(image_path)
    image_embeddings.append(embedding)

image_embeddings = torch.stack(image_embeddings)

# 2. 搜索功能
def search_images(query_text, top_k=10):
    query_embedding = clip_model.encode_text([query_text])
    similarities = query_embedding @ image_embeddings.t()
    top_indices = similarities.topk(k=top_k).indices[0]
    
    results = [image_database[i] for i in top_indices]
    return results

# 3. 使用示例
results = search_images("a cute dog playing in the park", top_k=5)
display_images(results)
```

### 6.2 项目2：多模态问答系统

**目标：** 结合BLIP实现视觉问答

```python
# VQA系统实现
class VisualQA:
    def __init__(self):
        self.blip_model = BLIP.from_pretrained('blip-vqa')
        
    def answer_question(self, image, question):
        # 编码
        image_embeds = self.blip_model.visual_encoder(image)
        question_tokens = self.tokenizer(question)
        
        # 生成答案
        answer = self.blip_model.generate_answer(
            image_embeds, 
            question_tokens,
            max_length=20
        )
        
        return self.tokenizer.decode(answer)
    
# 使用示例
vqa = VisualQA()
image = load_image('scene.jpg')
answer = vqa.answer_question(image, "How many people are in the image?")
print(f"答案: {answer}")
```

### 6.3 项目3：AI艺术生成器

**目标：** 使用DALL-E生成艺术作品

```python
class AIArtGenerator:
    def __init__(self):
        self.dalle = DALLE.from_pretrained('dalle-mini')
        
    def generate_art(self, prompt, num_images=4, style=None):
        if style:
            prompt = f"{prompt} in the style of {style}"
        
        images = self.dalle.generate(
            prompt, 
            num_images=num_images,
            temperature=0.9
        )
        
        return images
    
# 使用示例
generator = AIArtGenerator()
artworks = generator.generate_art(
    prompt="a futuristic city at sunset",
    style="cyberpunk",
    num_images=4
)

for i, art in enumerate(artworks):
    save_image(art, f'artwork_{i}.png')
```

---

## 第七章：最新研究和未来方向

### 7.1 扩散模型 (Diffusion Models)

**核心概念：**
- 前向过程：逐步添加噪声
- 反向过程：学习去噪
- 应用：Stable Diffusion, DALL-E 2

**简化实现：**
```python
class SimpleDiffusion(nn.Module):
    def __init__(self, num_steps=1000):
        super().__init__()
        self.num_steps = num_steps
        self.denoising_network = UNet()
        
    def forward_diffusion(self, x0, t):
        # 添加噪声
        noise = torch.randn_like(x0)
        alpha_t = self.get_alpha(t)
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return xt, noise
    
    def reverse_diffusion(self, xt, t):
        # 预测并去除噪声
        predicted_noise = self.denoising_network(xt, t)
        alpha_t = self.get_alpha(t)
        x_prev = (xt - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        return x_prev
```

### 7.2 大语言模型时代的多模态

**趋势：**
- GPT-4V: 视觉理解能力
- LLaVA: 大语言模型 + 视觉
- Flamingo: 少样本多模态学习

### 7.3 高效训练技术

**最新技术：**
- Flash Attention: 加速注意力计算
- Mixed Precision Training: 混合精度训练
- Gradient Checkpointing: 减少显存占用

---

## 附录

### A. 常用数据集

**计算机视觉：**
- MNIST: 手写数字识别
- CIFAR-10/100: 小图像分类
- ImageNet: 大规模图像分类
- COCO: 目标检测、分割、描述
- Visual Genome: 场景图理解

**自然语言处理：**
- WikiText: 语言建模
- GLUE: NLP任务集合
- SQuAD: 问答数据集

**多模态：**
- Conceptual Captions: 图像-文本对
- LAION-5B: 大规模图像-文本
- VQA: 视觉问答
- Flickr30k: 图像描述

### B. 学习资源

**论文：**
- Attention Is All You Need (Transformer)
- An Image is Worth 16x16 Words (ViT)
- Masked Autoencoders Are Scalable Vision Learners (MAE)
- Learning Transferable Visual Models From Natural Language Supervision (CLIP)
- Zero-Shot Text-to-Image Generation (DALL-E)

**在线课程：**
- Stanford CS231n: CNN for Visual Recognition
- Stanford CS224n: NLP with Deep Learning
- Fast.ai: Practical Deep Learning

**开源项目：**
- HuggingFace Transformers
- OpenAI CLIP
- Stable Diffusion

### C. 完整训练脚本示例

```python
# 完整的多模态训练流程
import torch
from torch.utils.data import DataLoader
from multimodal.clip.model import CLIP
from utils import Config, get_optimizer, get_scheduler

def train_clip():
    # 1. 配置
    config = Config.from_file('configs/clip_config.yaml')
    
    # 2. 模型
    model = CLIP(config.to_dict())
    model = model.cuda()
    
    # 3. 数据
    train_dataset = MultimodalDataset(config['data_path'], split='train')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 4. 优化器
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # 5. 训练循环
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            images = batch['images'].cuda()
            text_tokens = batch['text_tokens'].cuda()
            
            # 前向传播
            outputs = model(images, text_tokens)
            logits_per_image = outputs['logits_per_image']
            logits_per_text = outputs['logits_per_text']
            
            # 计算损失
            labels = torch.arange(len(images)).cuda()
            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 学习率调度
        scheduler.step()
        
        # 日志
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'clip_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train_clip()
```

---

## 结语

本教程系统地介绍了从经典深度学习模型到最新多模态大模型的完整知识体系。通过学习本教程，您应该能够：

1. ✅ 理解深度学习的核心原理
2. ✅ 掌握CNN、Transformer等经典架构
3. ✅ 实现和训练计算机视觉模型
4. ✅ 构建自然语言处理模型
5. ✅ 开发多模态应用系统

### 继续学习

- 阅读最新论文，关注arXiv和顶会
- 参与开源项目，贡献代码
- 实践更多项目，积累经验
- 加入社区，与同行交流

**祝您在多模态AI的学习之路上不断进步！** 🚀
