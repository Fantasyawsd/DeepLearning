# DeepLearning Models

这是一个深度学习领域各种经典模型的PyTorch复现项目。

## 已实现的模型

### 1. MAE (Masked Autoencoder)
- **论文**: "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
- **特点**: 自监督学习的视觉Transformer，通过掩码重建学习视觉表示
- **文件**: `models/mae.py`
- **配置**: `configs/mae_config.yaml`

### 2. BERT (Bidirectional Encoder Representations from Transformers)
- **论文**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (NAACL 2019)
- **特点**: 双向Transformer编码器，支持掩码语言模型和序列分类任务
- **文件**: `models/bert.py`
- **配置**: `configs/bert_config.yaml`

### 3. Swin Transformer
- **论文**: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021)
- **特点**: 分层视觉Transformer，使用滑动窗口机制
- **文件**: `models/swin_transformer.py`
- **配置**: `configs/swin_config.yaml`

## 项目结构

```
DeepLearning/
├── models/              # 模型实现
│   ├── __init__.py
│   ├── base.py         # 基础模型类
│   ├── mae.py          # MAE模型
│   ├── bert.py         # BERT模型
│   └── swin_transformer.py  # Swin Transformer模型
├── utils/              # 工具函数
│   ├── __init__.py
│   ├── config.py       # 配置管理
│   ├── logger.py       # 日志工具
│   └── metrics.py      # 评估指标
├── configs/            # 配置文件
│   ├── mae_config.yaml
│   ├── bert_config.yaml
│   └── swin_config.yaml
├── examples/           # 使用示例
│   ├── mae_example.py
│   ├── bert_example.py
│   └── swin_transformer_example.py
├── datasets/           # 数据集处理
├── train.py           # 训练脚本
├── requirements.txt   # 依赖包
├── setup.py          # 安装脚本
└── readme.md         # 项目说明
```

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
