# Getting Started Tutorial

This tutorial will walk you through the basics of using the Deep Learning Models Framework.

## Prerequisites

Make sure you have completed the installation steps from the [User Guide](USER_GUIDE.md#getting-started).

## Step 1: Import the Framework

```python
# Import the models
from models import MAE, BERT, SwinTransformer
from utils import Config

# Import PyTorch
import torch
import torch.nn as nn
```

## Step 2: Load a Configuration

```python
# Method 1: Load from YAML file
config = Config.from_file('configs/mae_config.yaml')

# Method 2: Create programmatically
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

print(f"Configuration: {config.to_dict()}")
```

## Step 3: Create and Use a Model

```python
# Create the model
model = MAE(config.to_dict())

# Get model information
print(f"Model name: {model.model_name}")
print(f"Parameters: {model.count_parameters():,}")

# Print model summary
model.summary()
```

## Step 4: Prepare Input Data

```python
# Create dummy input data
batch_size = 2
images = torch.randn(batch_size, 3, 224, 224)

print(f"Input shape: {images.shape}")
```

## Step 5: Forward Pass

```python
# Set model to evaluation mode
model.eval()

# Perform forward pass
with torch.no_grad():
    outputs = model(images)

# Examine outputs
print(f"Output keys: {list(outputs.keys())}")
print(f"Loss: {outputs['loss'].item():.6f}")
print(f"Predictions shape: {outputs['pred'].shape}")
```

## Step 6: Save and Load Models

```python
# Save model checkpoint
model.save_checkpoint('my_model_checkpoint.pth')

# Create new model and load checkpoint
new_model = MAE(config.to_dict())
new_model.load_checkpoint('my_model_checkpoint.pth')

print("Model checkpoint saved and loaded successfully!")
```

## Next Steps

1. **Explore Examples**: Check out the detailed examples in the `examples/` directory
2. **Read Model Guides**: Dive deep into specific models with the guides in `docs/models/`
3. **Train Your Own Model**: Use the training script or create custom training loops
4. **Experiment**: Modify configurations and try different model variants

## Common Patterns

### Working with Different Models

```python
# MAE for self-supervised learning
mae_config = Config.from_file('configs/mae_config.yaml')
mae_model = MAE(mae_config.to_dict())

# BERT for NLP tasks
bert_config = Config.from_file('configs/bert_config.yaml')
bert_model = BERT(bert_config.to_dict())

# Swin Transformer for image classification
swin_config = Config.from_file('configs/swin_config.yaml')
swin_model = SwinTransformer(swin_config.to_dict())
```

### Configuration Management

```python
# Modify configuration
config.set('learning_rate', 1e-4)
config.set('batch_size', 32)

# Get configuration values
lr = config.get('learning_rate', default=1e-3)
batch_size = config.get('batch_size', default=16)

# Convert to dictionary for model creation
model_config = config.to_dict()
```

### Training Setup

```python
# Create model
model = MAE(config.to_dict())

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Setup loss function (if needed)
criterion = nn.MSELoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs['loss']  # Many models provide loss directly
        loss.backward()
        optimizer.step()
```

This tutorial covers the basics. For advanced usage, refer to the specific model guides!