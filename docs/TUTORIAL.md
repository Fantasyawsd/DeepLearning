# 入门教程

本教程将引导您了解深度学习模型框架的基本使用方法。

## 前提条件

确保您已完成[用户指南](USER_GUIDE.md#开始使用)中的安装步骤。

## 步骤1：导入框架

```python
# 导入模型
from models import MAE, BERT, SwinTransformer
from utils import Config

# 导入PyTorch
import torch
import torch.nn as nn
```

## 步骤2：加载配置

```python
# 方法1：从YAML文件加载
config = Config.from_file('configs/mae_config.yaml')

# 方法2：以编程方式创建
config = Config({
    'model_name': 'mae',
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 768,
    'encoder_depth': 12,
    'encoder_num_heads': 12,
    'decoder_embed_dim': 512,
    'decoder_depth': 8,
    'decoder_num_heads': 16,
    'mask_ratio': 0.75
})

print(f"配置: {config.to_dict()}")
```

## 步骤3：创建和使用模型

```python
# 创建模型
model = MAE(config.to_dict())

# 获取模型信息
print(f"模型名称: {model.model_name}")
print(f"参数数量: {model.count_parameters():,}")

# 打印模型摘要
model.summary()
```

## 步骤4：准备输入数据

```python
# 创建虚拟输入数据
batch_size = 2
images = torch.randn(batch_size, 3, 224, 224)

print(f"输入形状: {images.shape}")
```

## 步骤5：前向传播

```python
# 将模型设置为评估模式
model.eval()

# 执行前向传播
with torch.no_grad():
    outputs = model(images)

# 检查输出
print(f"输出键: {list(outputs.keys())}")
print(f"损失: {outputs['loss'].item():.6f}")
print(f"预测形状: {outputs['pred'].shape}")
```

## 步骤6：保存和加载模型

```python
# 保存模型检查点
model.save_checkpoint('my_model_checkpoint.pth')

# 创建新模型并加载检查点
new_model = MAE(config.to_dict())
new_model.load_checkpoint('my_model_checkpoint.pth')

print("模型检查点已成功保存和加载！")
```

## 下一步

1. **探索示例**：查看`examples/`目录中的详细示例
2. **阅读模型指南**：深入了解`docs/models/`中的特定模型指南
3. **训练自己的模型**：使用训练脚本或创建自定义训练循环
4. **实验**：修改配置并尝试不同的模型变体

## 常见模式

### 使用不同的模型

```python
# MAE用于自监督学习
mae_config = Config.from_file('configs/mae_config.yaml')
mae_model = MAE(mae_config.to_dict())

# BERT用于NLP任务
bert_config = Config.from_file('configs/bert_config.yaml')
bert_model = BERT(bert_config.to_dict())

# Swin Transformer用于图像分类
swin_config = Config.from_file('configs/swin_config.yaml')
swin_model = SwinTransformer(swin_config.to_dict())
```

### 配置管理

```python
# 修改配置
config.set('learning_rate', 1e-4)
config.set('batch_size', 32)

# 获取配置值
lr = config.get('learning_rate', default=1e-3)
batch_size = config.get('batch_size', default=16)

# 转换为字典用于模型创建
model_config = config.to_dict()
```

### 训练设置

```python
# 创建模型
model = MAE(config.to_dict())

# 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 设置损失函数（如果需要）
criterion = nn.MSELoss()

# 训练循环
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs['loss']  # 许多模型直接提供损失
        loss.backward()
        optimizer.step()
```

This tutorial covers the basics. For advanced usage, refer to the specific model guides!