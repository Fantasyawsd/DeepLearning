# LeNet-5 模型使用指南

LeNet-5 是由 Yann LeCun 在1998年提出的经典卷积神经网络，是深度学习历史上的重要里程碑。本指南详细介绍 LeNet-5 的使用方法。

## 📖 模型简介

LeNet-5 是最早的成功的卷积神经网络之一，主要特点：

- **历史意义**: 深度学习的开创性工作
- **结构简单**: 2个卷积层 + 3个全连接层
- **轻量级**: 仅约60K参数
- **经典应用**: 手写数字识别、邮政编码识别

### 网络架构

```
输入(32×32) → Conv1(6@28×28) → Pool1(6@14×14) → Conv2(16@10×10) → Pool2(16@5×5) → FC1(120) → FC2(84) → FC3(10)
```

## 🚀 快速开始

### 1. 基本使用

```python
import torch
from model import LeNet5

# 创建模型 (MNIST配置)
config = {
    'num_classes': 10,
    'input_channels': 1,
    'input_size': 32
}
model = LeNet5(config)

# 前向传播
x = torch.randn(1, 1, 32, 32)  # 批次大小1，1通道，32×32图像
output = model(x)
print(f"输出形状: {output.shape}")  # [1, 10]
```

### 2. 使用配置文件训练

```bash
# 训练MNIST数据集
python train.py --config config.yaml

# 从检查点恢复训练
python train.py --config config.yaml --resume checkpoints/lenet/checkpoint_latest.pth
```

### 3. 模型测试

```bash
# 测试模型性能
python test.py --config config.yaml --checkpoint checkpoints/lenet/checkpoint_best.pth

# 可视化预测结果
python test.py --config config.yaml --checkpoint checkpoints/lenet/checkpoint_best.pth --visualize
```

## ⚙️ 配置说明

### 模型配置

```yaml
model:
  num_classes: 10        # 分类类别数
  input_channels: 1      # 输入通道数 (灰度图:1, RGB图:3)
  input_size: 32         # 输入图像尺寸
  dropout: 0.0           # Dropout比例 (原始LeNet不使用)
```

### 数据集支持

- **MNIST**: 手写数字识别 (28×28 → 32×32)
- **CIFAR-10**: 10类自然图像 (32×32)
- **CIFAR-100**: 100类自然图像 (32×32)

### 训练配置

```yaml
training:
  batch_size: 128
  epochs: 50
  learning_rate: 0.001
  weight_decay: 1e-4
  
  optimizer:
    type: "Adam"         # Adam, SGD, AdamW
    momentum: 0.9        # SGD使用
  
  scheduler:
    type: "StepLR"       # StepLR, CosineAnnealingLR
    step_size: 15
    gamma: 0.1
```

## 📊 性能基准

### MNIST数据集

| 配置 | 参数量 | 训练时间 | 测试精度 |
|------|--------|----------|----------|
| 标准LeNet-5 | 60K | ~5分钟 | 99%+ |
| 改进版(BatchNorm) | 60K | ~5分钟 | 99.3%+ |

### CIFAR-10数据集

| 配置 | 参数量 | 训练时间 | 测试精度 |
|------|--------|----------|----------|
| LeNet-5(RGB) | 62K | ~10分钟 | 65-70% |
| LeNet-5+数据增强 | 62K | ~12分钟 | 70-75% |

## 🔧 高级用法

### 1. 自定义模型

```python
from model import LeNet5

# 创建自定义配置
config = {
    'num_classes': 100,    # CIFAR-100
    'input_channels': 3,   # RGB图像
    'input_size': 32,
    'dropout': 0.2         # 添加Dropout防止过拟合
}

model = LeNet5(config)
print(f"模型参数量: {model.count_parameters():,}")
```

### 2. 加载预训练模型

```python
from load_pretrained import load_pretrained_lenet

# 加载预训练模型
model = load_pretrained_lenet('lenet5_mnist')

# 从检查点加载
from load_pretrained import create_lenet_from_checkpoint
model = create_lenet_from_checkpoint('checkpoints/lenet/checkpoint_best.pth')
```

### 3. 模型推理

```python
from test import LeNetTester

# 创建测试器
tester = LeNetTester('config.yaml', 'checkpoints/lenet/checkpoint_best.pth')

# 单张图像推理
result = tester.predict_single(image_tensor)
print(f"预测类别: {result['predicted_class_name']}")
print(f"置信度: {result['confidence']:.2f}")
```

### 4. 特征可视化

```python
import torch
import matplotlib.pyplot as plt

# 获取第一个卷积层的权重
conv1_weights = model.conv1.weight.data

# 可视化卷积核
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for i, ax in enumerate(axes.flat):
    if i < conv1_weights.shape[0]:
        # 显示第i个卷积核
        kernel = conv1_weights[i, 0]  # 第0个输入通道
        ax.imshow(kernel, cmap='gray')
        ax.set_title(f'卷积核 {i+1}')
        ax.axis('off')

plt.tight_layout()
plt.show()
```

## 🎯 训练技巧

### 1. 数据预处理

```python
# MNIST数据预处理
transforms.Compose([
    transforms.Resize(32),      # 调整到LeNet输入尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
])

# CIFAR数据预处理
transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 数据增强
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
])
```

### 2. 学习率调度

```python
# 阶梯衰减
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# 余弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 自适应调整
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
```

### 3. 防止过拟合

```python
# 1. 添加Dropout
config['dropout'] = 0.2

# 2. 数据增强
data_config = {
    'augmentation': {
        'enabled': True,
        'horizontal_flip': True,
        'rotation': 10
    }
}

# 3. 正则化
optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
```

## 📈 模型分析

### 1. 参数统计

```python
# 总参数量
total_params = model.count_parameters()
print(f"总参数量: {total_params:,}")

# 各层参数量
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} 参数")
```

### 2. 计算复杂度

```python
from torchprofile import profile_macs

# 计算FLOPs
input_tensor = torch.randn(1, 1, 32, 32)
macs = profile_macs(model, input_tensor)
print(f"FLOPs: {macs:,}")
```

### 3. 推理速度

```python
import time

model.eval()
input_tensor = torch.randn(100, 1, 32, 32)

# 预热
for _ in range(10):
    _ = model(input_tensor)

# 测试推理速度
start_time = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(input_tensor)
end_time = time.time()

avg_time = (end_time - start_time) / 100 / 100  # 每张图像时间
print(f"平均推理时间: {avg_time*1000:.2f}ms/图像")
```

## 🔍 故障排除

### 常见问题

1. **输入尺寸错误**
   ```python
   # 确保输入尺寸为32x32
   assert input.shape[-2:] == (32, 32), f"输入尺寸应为32x32，当前为{input.shape[-2:]}"
   ```

2. **类别数不匹配**
   ```python
   # 检查配置中的类别数
   assert config['num_classes'] == len(class_names), "类别数与标签数不匹配"
   ```

3. **GPU内存不足**
   ```python
   # 减小批次大小
   config['training']['batch_size'] = 64  # 从128减小到64
   ```

### 调试技巧

```python
# 1. 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: 梯度范数 {param.grad.norm():.6f}")

# 2. 监控损失
if torch.isnan(loss):
    print("警告: 损失为NaN，检查学习率设置")

# 3. 可视化激活
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__} 输出形状: {output.shape}")

# 注册钩子
model.conv1.register_forward_hook(hook_fn)
model.conv2.register_forward_hook(hook_fn)
```

## 📚 参考资料

- **原始论文**: [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- **作者**: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner
- **发表年份**: 1998
- **应用领域**: 文档识别、手写数字识别

## 🎓 学习建议

1. **理解原理**: LeNet是理解CNN的最佳起点
2. **动手实践**: 在不同数据集上训练模型
3. **对比分析**: 与现代CNN架构对比学习
4. **可视化分析**: 观察卷积核和特征图的变化
5. **性能优化**: 尝试不同的优化技巧

LeNet-5虽然简单，但包含了卷积神经网络的核心思想，是深度学习入门的理想选择！