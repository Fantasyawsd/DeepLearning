# DeepLearning Models Framework | 深度学习模型框架

这是一个全面的深度学习模型框架，包含计算机视觉和自然语言处理领域的经典模型PyTorch实现，按照领域和任务分类组织。

*A comprehensive deep learning models framework with PyTorch implementations of classic models in computer vision and natural language processing, organized by domain and task.*

## 🏗️ 项目结构 | Project Structure

本项目采用**领域驱动**的组织方式，按照应用领域和具体任务分类：

```
DeepLearning/
├── computer_vision/           # 计算机视觉
│   ├── image_classification/  # 图像分类
│   │   ├── cnn/              # CNN系列
│   │   │   ├── lenet/        # LeNet-5 (1998) ✅
│   │   │   ├── alexnet/      # AlexNet (2012)
│   │   │   ├── vgg/          # VGG (2014)
│   │   │   └── resnet/       # ResNet系列 (2015) ✅
│   │   └── transformer/      # Transformer系列
│   │       ├── vit/          # Vision Transformer (2020) ✅
│   │       ├── swin_transformer/ # Swin Transformer (2021)
│   │       └── mae/          # Masked Autoencoder (2022) ✅
│   └── object_detection/     # 目标检测
│       ├── rcnn_series/      # R-CNN系列
│       │   ├── rcnn/         # R-CNN (2014)
│       │   ├── fast_rcnn/    # Fast R-CNN (2015)
│       │   ├── faster_rcnn/  # Faster R-CNN (2015)
│       │   └── mask_rcnn/    # Mask R-CNN (2017)
│       └── yolo_series/      # YOLO系列
│           ├── yolov1/       # YOLOv1 (2016) ✅
│           ├── yolov3/       # YOLOv3 (2018)
│           ├── yolov5/       # YOLOv5 (2020)
│           └── yolov8/       # YOLOv8 (2023)
├── nlp/                      # 自然语言处理
│   └── language_models/      # 语言模型
│       ├── bert/             # BERT (2018)
│       ├── gpt_series/       # GPT系列
│       │   ├── gpt/          # GPT-1 (2018) ✅
│       │   ├── gpt2/         # GPT-2 (2019)
│       │   └── gpt3/         # GPT-3 (2020)
│       └── llm/              # 大语言模型
│           ├── llama/        # LLaMA (2023)
│           └── chatglm/      # ChatGLM (2023)
├── utils/                    # 通用工具
├── docs/                     # 文档
└── configs/                  # 配置文件
```

## 🚀 核心特性 | Key Features

### 📦 完整模型包
每个模型都包含完整的组件：
- **模型定义** (`model.py`) - 完整架构实现
- **数据处理** (`dataset.py`) - 数据加载和预处理
- **训练脚本** (`train.py`) - 完整训练流程
- **测试工具** (`test.py`) - 推理和评估
- **配置文件** (`config.yaml`) - 可调参数配置
- **预训练集成** (`load_pretrained.py`) - HuggingFace模型加载
- **完整文档** (`README.md`) - 使用指南和示例

### 🎯 统一设计
- **统一基类**: 所有模型继承自 `BaseModel`
- **一致接口**: 标准化的训练、推理和配置接口
- **模块化设计**: 可复用的组件和工具函数

### 🌐 中文优化
- **完整中文文档**: 针对中文用户优化的说明文档
- **中文注释**: 详细的代码中文注释
- **本土化配置**: 适合中文环境的默认设置

## 📖 Documentation | 文档

**主要指南 | Main Guides:**
- **[用户指南 | User Guide](docs/USER_GUIDE.md)** - 完整框架使用教程
- **[入门教程 | Tutorial](docs/TUTORIAL.md)** - 逐步教程和代码示例
- **[文档索引 | Documentation Index](docs/README.md)** - 文档导航中心

**模型文档 | Model Documentation:**
- **[MAE 完整指南](computer_vision/image_classification/transformer/mae/README.md)** - 自监督视觉学习
- **[ViT 指南](computer_vision/image_classification/transformer/vit/README.md)** - Vision Transformer
- **[ResNet 指南](computer_vision/image_classification/cnn/resnet/README.md)** - 残差网络
- **[YOLOv1 指南](computer_vision/object_detection/yolo_series/yolov1/README.md)** - 实时目标检测
- **[GPT 指南](nlp/language_models/gpt_series/gpt/README.md)** - 生成式语言模型

## 🏆 已实现模型 | Implemented Models

### 🖼️ 计算机视觉 | Computer Vision

#### 图像分类 | Image Classification

**CNN 系列:**
- ✅ **LeNet-5** (1998) - 卷积神经网络先驱，手写数字识别
- ✅ **ResNet** (2015) - 残差网络，解决深度网络梯度消失问题
  - ResNet-18/34/50/101/152
  - Pre-activation ResNet

**Transformer 系列:**
- ✅ **ViT** (2020) - Vision Transformer，将Transformer引入视觉
  - ViT-Tiny/Small/Base/Large/Huge
  - DeiT (Data-efficient Image Transformers)
- ✅ **MAE** (2022) - 掩码自编码器，自监督视觉学习
  - 完整编码器-解码器架构
  - HuggingFace预训练模型集成

#### 目标检测 | Object Detection

**YOLO 系列:**
- ✅ **YOLOv1** (2016) - 首个端到端实时目标检测系统
  - 完整YOLO架构和损失函数
  - 非极大值抑制 (NMS)

### 💬 自然语言处理 | Natural Language Processing

**语言模型 | Language Models:**
- ✅ **GPT** (2018) - 生成式预训练Transformer
  - 自回归语言建模
  - 文本生成和序列分类
  - GPT-Small/Medium/Large/XL变体

## 🚀 快速开始 | Quick Start

### 1. 环境安装 | Installation

```bash
# 克隆项目
git clone https://github.com/Fantasyawsd/DeepLearning.git
cd DeepLearning

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

### 2. 运行示例 | Run Examples

```bash
# MAE自监督学习示例
cd computer_vision/image_classification/transformer/mae
python train.py --config config.yaml

# ViT图像分类示例  
cd computer_vision/image_classification/transformer/vit
python train.py --config config.yaml

# GPT文本生成示例
cd nlp/language_models/gpt_series/gpt
python train.py --config config.yaml

# YOLOv1目标检测示例
cd computer_vision/object_detection/yolo_series/yolov1
python train.py --config config.yaml
```

### 3. 加载预训练模型 | Load Pretrained Models

```python
# MAE预训练模型
from computer_vision.image_classification.transformer.mae.load_pretrained import load_pretrained_mae
model = load_pretrained_mae('mae-base')

# ViT预训练模型
from computer_vision.image_classification.transformer.vit.load_pretrained import load_pretrained_vit
model = load_pretrained_vit('vit_base_patch16_224')
```

## 📊 模型性能对比 | Model Performance

| 模型 | 任务 | 数据集 | 参数量 | 性能指标 |
|------|------|--------|--------|---------|
| MAE-Base | 自监督学习 | ImageNet | 87M | 83.6% (微调后) |
| ViT-Base | 图像分类 | ImageNet | 86M | 84.5% Top-1 |
| ResNet-50 | 图像分类 | ImageNet | 25M | 76.2% Top-1 |
| YOLOv1 | 目标检测 | PASCAL VOC | 45M | 63.4 mAP |
| GPT-Small | 语言模型 | WebText | 117M | 18.3 PPL |

## 🛠️ 开发路线图 | Development Roadmap

### 🎯 即将完成 | Coming Soon
- [ ] **AlexNet, VGG** - 经典CNN架构
- [ ] **Faster R-CNN, Mask R-CNN** - 两阶段目标检测
- [ ] **YOLOv3/v5/v8** - YOLO系列演进
- [ ] **GPT-2/GPT-3** - GPT系列扩展
- [ ] **LLaMA, ChatGLM** - 现代大语言模型

### 🔮 未来计划 | Future Plans
- [ ] **扩散模型** (Diffusion Models)
- [ ] **多模态模型** (CLIP, DALL-E)
- [ ] **强化学习** (DQN, PPO)
- [ ] **图神经网络** (GCN, GraphSAGE)

## 🤝 贡献指南 | Contributing

欢迎贡献代码！请遵循以下流程：

1. **Fork** 项目
2. 创建特性分支: `git checkout -b feature/AmazingFeature`
3. 提交更改: `git commit -m 'Add some AmazingFeature'`
4. 推送分支: `git push origin feature/AmazingFeature`
5. 提交 **Pull Request**

### 贡献要求 | Contribution Requirements
- 代码符合项目规范
- 添加完整的文档和示例
- 包含单元测试
- 提供中文文档

## 📄 许可证 | License

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢 | Acknowledgments

感谢以下论文作者和开源项目：
- MAE: He et al. (Meta AI)
- ViT: Dosovitskiy et al. (Google)
- ResNet: He et al. (Microsoft)
- YOLO: Redmon et al.
- GPT: Radford et al. (OpenAI)
- HuggingFace Transformers
- PyTorch团队

---

**⭐ 如果这个项目对你有帮助，请给个星标！**

## 安装

1. 克隆项目
```bash
git clone https://github.com/Fantasyawsd/DeepLearning.git
cd DeepLearning
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 安装包
```bash
pip install -e .
```

## 使用方法

### 📚 详细文档

查看完整使用指南和教程：
- **[用户指南](docs/USER_GUIDE.md)** - 完整的框架使用教程
- **[MAE 完整指南](docs/models/MAE_GUIDE.md)** - MAE模型详细使用说明
- **[BERT 完整指南](docs/models/BERT_GUIDE.md)** - BERT模型详细使用说明
- **[Swin Transformer 完整指南](docs/models/SWIN_TRANSFORMER_GUIDE.md)** - Swin Transformer模型详细使用说明

### 1. 快速体验

运行示例脚本：
```bash
# MAE示例
python examples/mae_example.py

# BERT示例
python examples/bert_example.py

# Swin Transformer示例
python examples/swin_transformer_example.py
```

### 2. 训练模型

使用配置文件训练模型：
```bash
# 训练MAE
python train.py --config configs/mae_config.yaml --output_dir outputs/mae

# 训练BERT
python train.py --config configs/bert_config.yaml --output_dir outputs/bert

# 训练Swin Transformer
python train.py --config configs/swin_config.yaml --output_dir outputs/swin
```

### 3. 自定义使用

```python
from models import MAE, BERT, SwinTransformer
from utils import Config

# 加载配置
config = Config.from_file('configs/mae_config.yaml')

# 创建模型
model = MAE(config.to_dict())

# 模型信息
model.summary()

# 前向传播
import torch
x = torch.randn(1, 3, 224, 224)
output = model(x)
```

## 模型特性

### 通用特性
- 统一的基础模型类 (`BaseModel`)
- 完整的检查点保存/加载机制
- 参数冻结/解冻功能
- 模型信息统计

### MAE特性
- 完整的编码器-解码器架构
- 可配置的掩码比例
- 位置编码支持
- 重建损失计算

### BERT特性
- 多种任务支持（MLM、分类等）
- 注意力机制可视化
- 灵活的输入格式
- 预训练权重兼容

### Swin Transformer特性
- 分层特征提取
- 滑动窗口注意力
- 相对位置编码
- 高效的计算复杂度

## 依赖

- Python >= 3.8
- PyTorch >= 1.12.0
- transformers >= 4.20.0
- einops >= 0.4.1
- timm >= 0.6.7
- 其他依赖见 `requirements.txt`

## 贡献

欢迎提交PR和Issue！请确保：
1. 代码符合项目规范
2. 添加必要的测试
3. 更新相关文档

## 许可证

MIT License

## 致谢

感谢以下论文的作者：
- MAE: He et al., "Masked Autoencoders Are Scalable Vision Learners"
- BERT: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Swin Transformer: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
