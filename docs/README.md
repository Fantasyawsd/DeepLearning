# 文档索引

欢迎来到深度学习模型框架文档！本部分提供了使用框架及其模型的全面指南和教程。

## 📚 文档结构

### 主要指南
- **[用户指南](USER_GUIDE.md)** - 框架入门完整教程
- **[入门教程](TUTORIAL.md)** - 带代码示例的逐步教程
- **[安装指南](#安装)** - 逐步安装说明
- **[训练指南](#训练)** - 如何使用框架训练模型

### 模型特定指南
- **[MAE指南](models/MAE_GUIDE.md)** - 掩码自编码器完整指南

**注意：** 项目已重构为域分类结构。每个模型在其目录下都有详细的README文档：
- 计算机视觉模型：`computer_vision/` 目录下
- 自然语言处理模型：`nlp/` 目录下

## 🚀 快速开始

1. **安装**：按照[用户指南](USER_GUIDE.md#开始使用)进行安装
2. **基本用法**：尝试[examples/](../examples/)目录中的示例
3. **模型训练**：使用训练脚本或实现自定义训练循环
4. **高级用法**：探索模型特定指南以了解高级功能

## 📖 各指南内容

### 用户指南
- 安装和设置
- 项目结构概述
- 基本使用模式
- 训练说明
- 高级功能
- 故障排除

### 模型指南
每个模型指南包括：
- 模型概述和关键概念
- 配置选项
- 使用示例（从基础到高级）
- 训练说明
- 可视化技术
- 性能优化提示
- 常见问题和解决方案

## 🔧 配置管理

所有模型使用位于[`configs/`](../configs/)中的YAML配置文件：
- `mae_config.yaml` - MAE掩码自编码器配置
- `vit_config.yaml` - Vision Transformer配置
- `lenet_config.yaml` - LeNet配置
- `resnet_config.yaml` - ResNet配置
- `gpt_config.yaml` - GPT配置
- `yolo_config.yaml` - YOLOv1配置

## 💡 示例和教程

查看[`examples/`](../examples/)目录获取工作代码：
- `mae_example.py` - MAE使用示例

更多示例请查看各模型目录下的具体实现。

## 🛠️ 开发和贡献

### 项目结构
```
docs/
├── README.md                    # 本文件
├── USER_GUIDE.md               # 主用户指南
└── models/                     # 模型特定指南
    ├── MAE_GUIDE.md           # MAE文档
    ├── BERT_GUIDE.md          # BERT文档
    └── SWIN_TRANSFORMER_GUIDE.md # Swin Transformer文档
```

### 添加新文档
1. 在适当目录中创建新指南
2. 遵循现有格式和结构
3. 包含实用示例和代码片段
4. 更新此索引以引用新指南

## 📞 获取帮助

1. **查看相关指南**了解您的模型或用例
2. **查看示例**在`examples/`目录中
3. **检查常见问题**在故障排除部分
4. 如需额外帮助，**在GitHub上开issue**

## 🔗 外部资源

### 论文
- **MAE**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- **Swin Transformer**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

### 官方实现
- **MAE**: [Facebook Research MAE](https://github.com/facebookresearch/mae)
- **BERT**: [Google Research BERT](https://github.com/google-research/bert)
- **Swin Transformer**: [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer)

---

*本文档持续更新。最后更新：2024年*