# Deep Learning Models Framework - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Project Structure](#project-structure)
4. [Model Usage](#model-usage)
5. [Training Your Models](#training-your-models)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Introduction

Welcome to the Deep Learning Models Framework! This project provides PyTorch implementations of three state-of-the-art deep learning models:

- **MAE (Masked Autoencoder)** - Self-supervised learning for computer vision
- **BERT** - Bidirectional transformers for natural language processing  
- **Swin Transformer** - Hierarchical vision transformer for image classification

All models are built on a unified framework with consistent APIs, making it easy to experiment with different architectures.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12.0 or higher
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Fantasyawsd/DeepLearning.git
   cd DeepLearning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   python -c "from models import MAE, BERT, SwinTransformer; print('Installation successful!')"
   ```

### Quick Start

Run the example scripts to verify everything works:

```bash
# Test MAE
python examples/mae_example.py

# Test BERT
python examples/bert_example.py

# Test Swin Transformer
python examples/swin_transformer_example.py
```

## Project Structure

```
DeepLearning/
├── models/                    # Model implementations
│   ├── base.py               # Base model class
│   ├── mae.py                # MAE implementation
│   ├── bert.py               # BERT implementation
│   └── swin_transformer.py   # Swin Transformer implementation
├── utils/                    # Utility modules
│   ├── config.py            # Configuration management
│   ├── logger.py            # Logging utilities
│   └── metrics.py           # Evaluation metrics
├── configs/                  # Model configurations
│   ├── mae_config.yaml      # MAE configuration
│   ├── bert_config.yaml     # BERT configuration
│   └── swin_config.yaml     # Swin Transformer configuration
├── examples/                 # Usage examples
├── docs/                     # Documentation
├── datasets/                 # Dataset utilities
├── train.py                  # Training script
└── requirements.txt          # Dependencies
```

## Model Usage

### Basic Usage Pattern

All models follow the same usage pattern:

```python
from models import ModelName
from utils import Config

# 1. Load configuration
config = Config.from_file('configs/model_config.yaml')
# OR create config programmatically
config = Config({
    'model_name': 'model_name',
    'param1': value1,
    'param2': value2,
})

# 2. Create model
model = ModelName(config.to_dict())

# 3. Use the model
model.eval()  # Set to evaluation mode
outputs = model(inputs)

# 4. Get model information
model.summary()
print(f"Model parameters: {model.count_parameters()}")
```

### Configuration Management

The framework uses YAML configuration files for easy model customization:

```python
from utils import Config

# Load from file
config = Config.from_file('configs/mae_config.yaml')

# Create programmatically
config = Config({
    'model_name': 'mae',
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 768
})

# Access values
embed_dim = config.get('embed_dim', default_value)

# Convert to dictionary
model_config = config.to_dict()
```

### Model Checkpoints

All models support checkpoint saving and loading:

```python
# Save checkpoint
model.save_checkpoint('path/to/checkpoint.pth')

# Load checkpoint
model.load_checkpoint('path/to/checkpoint.pth')

# Save only state dict
torch.save(model.state_dict(), 'model_weights.pth')

# Load state dict
model.load_state_dict(torch.load('model_weights.pth'))
```

## Training Your Models

### Using the Training Script

The framework includes a comprehensive training script:

```bash
# Basic training
python train.py --config configs/mae_config.yaml --output_dir outputs/mae_experiment

# Advanced training with custom parameters
python train.py \
    --config configs/bert_config.yaml \
    --output_dir outputs/bert_experiment \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --device cuda \
    --save_every 10
```

### Custom Training Loop

For more control, implement your own training loop:

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from models import MAE
from utils import Config

# Setup
config = Config.from_file('configs/mae_config.yaml')
model = MAE(config.to_dict())
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    # Save checkpoint
    if epoch % 10 == 0:
        model.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
```

## Advanced Usage

### Parameter Freezing

Control which parts of the model to train:

```python
# Freeze encoder, train only decoder
model.freeze_encoder()

# Freeze all parameters except classification head
model.freeze_all()
model.unfreeze_classifier()

# Custom freezing
for name, param in model.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False
```

### Model Inspection

Get detailed information about your models:

```python
# Model summary
model.summary()

# Parameter count
total_params = model.count_parameters()
trainable_params = model.count_parameters(only_trainable=True)

# Layer information
for name, module in model.named_modules():
    print(f"{name}: {module}")

# Parameter information
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad: {param.requires_grad}")
```

### Multi-GPU Training

Use DataParallel or DistributedDataParallel:

```python
import torch.nn as nn

# DataParallel (simpler but less efficient)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.cuda()

# DistributedDataParallel (recommended for multi-GPU)
# See PyTorch DDP documentation for detailed setup
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Model not learning:**
   - Check learning rate (try 1e-3, 1e-4, 1e-5)
   - Verify data preprocessing
   - Check loss function and metrics

3. **Import errors:**
   - Ensure all dependencies are installed
   - Check Python path and package installation

4. **Configuration errors:**
   - Validate YAML syntax
   - Check parameter names and values
   - Use Config.validate() if available

### Performance Optimization

1. **Memory optimization:**
   ```python
   # Enable mixed precision
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

2. **Gradient accumulation:**
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

### Getting Help

1. Check the model-specific documentation in `docs/models/`
2. Review the example scripts in `examples/`
3. Create an issue on GitHub with:
   - Full error message
   - Code snippet that reproduces the issue
   - Environment details (Python version, PyTorch version, etc.)

## Next Steps

- Read the individual model guides in `docs/models/`
- Explore the example scripts in `examples/`
- Try training on your own data
- Experiment with different configurations
- Contribute improvements back to the project!