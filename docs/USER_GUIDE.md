# 深度学习模型框架 - 用户指南

## 目录
1. [介绍](#介绍)
2. [开始使用](#开始使用)
3. [项目结构](#项目结构)
4. [模型使用](#模型使用)
5. [训练模型](#训练模型)
6. [高级用法](#高级用法)
7. [故障排除](#故障排除)

## 介绍

欢迎使用深度学习模型框架！本项目提供了三个前沿深度学习模型的PyTorch实现：

- **MAE (掩码自编码器)** - 计算机视觉的自监督学习
- **BERT** - 自然语言处理的双向Transformer
- **Swin Transformer** - 图像分类的分层视觉Transformer

所有模型都构建在统一的框架上，具有一致的API，便于试验不同的架构。

## 开始使用

### 环境要求

- Python 3.8 或更高版本
- PyTorch 1.12.0 或更高版本
- CUDA兼容的GPU（推荐用于训练）

### 安装

1. **克隆仓库：**
   ```bash
   git clone https://github.com/Fantasyawsd/DeepLearning.git
   cd DeepLearning
   ```

2. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

3. **以开发模式安装包：**
   ```bash
   pip install -e .
   ```

4. **验证安装：**
   ```bash
   python -c "from models import MAE, BERT, SwinTransformer; print('安装成功！')"
   ```

### 快速开始

运行示例脚本验证一切正常：

```bash
# 测试MAE
python examples/mae_example.py

# 测试BERT
python examples/bert_example.py

# 测试Swin Transformer
python examples/swin_transformer_example.py
```

## 项目结构

```
DeepLearning/
├── models/                    # 模型实现
│   ├── base.py               # 基础模型类
│   ├── mae.py                # MAE实现
│   ├── bert.py               # BERT实现
│   └── swin_transformer.py   # Swin Transformer实现
├── utils/                    # 工具模块
│   ├── config.py            # 配置管理
│   ├── logger.py            # 日志工具
│   └── metrics.py           # 评估指标
├── configs/                  # 模型配置
│   ├── mae_config.yaml      # MAE配置
│   ├── bert_config.yaml     # BERT配置
│   └── swin_config.yaml     # Swin Transformer配置
├── examples/                 # 使用示例
├── docs/                     # 文档
├── datasets/                 # 数据集工具
├── train.py                  # 训练脚本
└── requirements.txt          # 依赖
```

## 模型使用

### 基本使用模式

所有模型都遵循相同的使用模式：

```python
from models import ModelName
from utils import Config

# 1. 加载配置
config = Config.from_file('configs/model_config.yaml')
# 或者以编程方式创建配置
config = Config({
    'model_name': 'model_name',
    'param1': value1,
    'param2': value2,
})

# 2. 创建模型
model = ModelName(config.to_dict())

# 3. 使用模型
model.eval()  # 设置为评估模式
outputs = model(inputs)

# 4. 获取模型信息
model.summary()
print(f"模型参数数量: {model.count_parameters()}")
```

### 配置管理

框架使用YAML配置文件来方便模型自定义：

```python
from utils import Config

# 从文件加载
config = Config.from_file('configs/mae_config.yaml')

# 以编程方式创建
config = Config({
    'model_name': 'mae',
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 768
})

# 访问值
embed_dim = config.get('embed_dim', default_value)

# 转换为字典
model_config = config.to_dict()
```

### 模型检查点

所有模型都支持检查点保存和加载：

```python
# 保存检查点
model.save_checkpoint('path/to/checkpoint.pth')

# 加载检查点
model.load_checkpoint('path/to/checkpoint.pth')

# 仅保存状态字典
torch.save(model.state_dict(), 'model_weights.pth')

# 加载状态字典
model.load_state_dict(torch.load('model_weights.pth'))
```

## 训练模型

### 使用训练脚本

框架包含一个全面的训练脚本：

```bash
# 基础训练
python train.py --config configs/mae_config.yaml --output_dir outputs/mae_experiment

# 使用自定义参数的高级训练
python train.py \
    --config configs/bert_config.yaml \
    --output_dir outputs/bert_experiment \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --device cuda \
    --save_every 10
```

### 自定义训练循环

要获得更多控制，实现自己的训练循环：

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from models import MAE
from utils import Config

# 设置
config = Config.from_file('configs/mae_config.yaml')
model = MAE(config.to_dict())
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 训练循环
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(data)
        loss = outputs['loss']
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'轮次 {epoch}, 批次 {batch_idx}, 损失: {loss.item():.6f}')
    
    # 保存检查点
    if epoch % 10 == 0:
        model.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
```

## 高级用法

### 参数冻结

控制模型的哪些部分需要训练：

```python
# 冻结编码器，仅训练解码器
model.freeze_encoder()

# 冻结所有参数，除了分类头
model.freeze_all()
model.unfreeze_classifier()

# 自定义冻结
for name, param in model.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False
```

### 模型检查

获取模型的详细信息：

```python
# 模型摘要
model.summary()

# 参数计数
total_params = model.count_parameters()
trainable_params = model.count_parameters(only_trainable=True)

# 层信息
for name, module in model.named_modules():
    print(f"{name}: {module}")

# 参数信息
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad: {param.requires_grad}")
```

### 多GPU训练

使用DataParallel或DistributedDataParallel：

```python
import torch.nn as nn

# DataParallel（更简单但效率较低）
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.cuda()

# DistributedDataParallel（推荐用于多GPU）
# 详细设置请参见PyTorch DDP文档
```

## 故障排除

### 常见问题

1. **CUDA内存不足：**
   - 减少批次大小
   - 使用梯度累积
   - 启用混合精度训练

2. **模型不学习：**
   - 检查学习率（尝试1e-3、1e-4、1e-5）
   - 验证数据预处理
   - 检查损失函数和指标

3. **导入错误：**
   - 确保所有依赖都已安装
   - 检查Python路径和包安装

4. **配置错误：**
   - 验证YAML语法
   - 检查参数名称和值
   - 如果可用，使用Config.validate()

### 性能优化

1. **内存优化：**
   ```python
   # 启用混合精度
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   model.train()
   
   for data in dataloader:
       optimizer.zero_grad()
       with autocast():
           outputs = model(data)
           loss = outputs['loss']
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

2. **梯度累积：**
   ```python
   accumulation_steps = 4
   
   for i, (data, _) in enumerate(dataloader):
       outputs = model(data)
       loss = outputs['loss'] / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### 获取帮助

1. 查看`docs/models/`中的模型特定文档
2. 查看`examples/`中的示例脚本
3. 在GitHub上创建Issue，包含：
   - 完整的错误信息
   - 能重现问题的代码片段
   - 环境详情（Python版本、PyTorch版本等）

## 下一步

- 阅读`docs/models/`中的各个模型指南
- 探索`examples/`中的示例脚本
- 尝试在自己的数据上训练
- 试验不同的配置
- Contribute improvements back to the project!