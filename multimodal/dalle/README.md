# DALL-E - Text-to-Image Generation

## 简介

DALL-E是OpenAI开发的文本到图像生成模型，能够根据文本描述生成高质量、创意性的图像。

## 核心特性

- **文本条件生成**: 根据自然语言描述生成图像
- **创意组合**: 能够组合不同概念创造新图像
- **高质量输出**: 生成逼真或艺术风格的图像
- **零样本泛化**: 生成训练中未见过的概念组合

## 模型架构

```
文本 → Text Encoder → 文本token嵌入
→ Transformer (自回归生成) → 图像token序列
→ dVAE Decoder → 生成图像 (256x256)
```

### 关键组件

1. **dVAE (discrete VAE)**: 将图像编码为离散token
2. **Text Encoder**: 编码文本描述
3. **Transformer**: 自回归生成图像token序列

## 快速开始

### 基本使用

```python
from multimodal.dalle.model import DALLE
from utils import Config

# 加载配置
config = Config.from_file('multimodal/dalle/config.yaml')

# 创建模型
model = DALLE(config.to_dict())

# 文本到图像生成
text_prompt = "a painting of a cat wearing a hat in the style of Van Gogh"
generated_images = model.generate(text_prompt, num_images=4)

# 保存结果
for i, img in enumerate(generated_images):
    save_image(img, f'dalle_output_{i}.png')
```

## 应用示例

### 1. 艺术创作

```python
prompts = [
    "a surrealist painting of melting clocks in a desert",
    "abstract art with geometric shapes in vibrant colors",
    "a photorealistic portrait in the style of Rembrandt"
]

for prompt in prompts:
    images = model.generate(prompt, num_images=4, temperature=0.9)
    display_images(images, title=prompt)
```

### 2. 概念组合

```python
# 组合不同的概念
prompt = "an astronaut riding a horse on the moon"
images = model.generate(prompt, num_images=8)
```

### 3. 风格迁移

```python
# 指定艺术风格
object_desc = "a peaceful mountain landscape"
styles = ["impressionist", "cubist", "art deco", "anime"]

for style in styles:
    prompt = f"{object_desc} in {style} style"
    image = model.generate(prompt, num_images=1)
    save_image(image, f'{style}_landscape.png')
```

## 训练

### 数据准备

需要图像-文本对数据集：
```python
from multimodal.dalle.dataset import ImageTextDataset

dataset = ImageTextDataset(
    data_dir='path/to/data',
    image_size=256,
    text_max_length=256
)
```

### 训练脚本

```bash
python multimodal/dalle/train.py \
    --config multimodal/dalle/config.yaml \
    --output_dir outputs/dalle \
    --batch_size 64 \
    --epochs 100
```

### 两阶段训练

1. **dVAE预训练**:
```bash
python multimodal/dalle/train_dvae.py \
    --data_dir path/to/images \
    --output_dir outputs/dvae
```

2. **DALL-E主模型训练**:
```bash
python multimodal/dalle/train.py \
    --dvae_checkpoint outputs/dvae/best.pth \
    --config multimodal/dalle/config.yaml
```

## 生成参数调优

### Temperature

控制生成的随机性：
```python
# 低temperature：更确定性的输出
images = model.generate(prompt, temperature=0.5)

# 高temperature：更多样化的输出
images = model.generate(prompt, temperature=1.2)
```

### Top-k和Top-p采样

```python
# Top-k采样：只从概率最高的k个token中采样
images = model.generate(prompt, top_k=256)

# Top-p (nucleus)采样：累积概率达到p的token集合
images = model.generate(prompt, top_p=0.95)
```

## 性能优化

### 批量生成

```python
# 同时生成多个不同提示的图像
prompts = ["cat", "dog", "bird", "fish"]
batch_images = model.batch_generate(prompts, num_images_per_prompt=4)
```

### 混合精度

```python
# 使用FP16加速生成
model = model.half()
images = model.generate(prompt)
```

## 实用技巧

### Prompt工程

好的提示词技巧：
- 具体描述：包含主体、风格、细节
- 使用形容词：增加视觉细节
- 指定艺术家风格：引导特定风格
- 添加质量词：如"high quality", "detailed"

示例：
```python
# 普通提示
prompt1 = "a house"

# 优化后的提示
prompt2 = "a cozy cottage with a thatched roof, surrounded by colorful flowers, in the style of Thomas Kinkade, highly detailed, warm lighting"
```

## 参考文献

```bibtex
@article{ramesh2021zero,
  title={Zero-shot text-to-image generation},
  author={Ramesh, Aditya and Pavlov, Mikhail and Goh, Gabriel and others},
  journal={ICML},
  year={2021}
}
```

## 许可证

MIT License
