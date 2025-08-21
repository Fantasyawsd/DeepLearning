# BERT (来自Transformer的双向编码器表示) - 完整指南

## 概述

BERT是一个双向transformer模型，通过引入transformer的双向训练革命性地改变了自然语言处理。发表于"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"（NAACL 2019），在许多NLP基准测试中达到了最先进的结果。

## 核心概念

### 架构
- **双向Transformer**：同时双向处理文本
- **多层架构**：Transformer编码器层的堆叠
- **注意力机制**：多头自注意力用于捕获关系
- **位置嵌入**：学习的位置表示

### 训练方法
1. **掩码语言建模（MLM）**：从上下文预测掩码标记
2. **下一句预测（NSP）**：判断两个句子是否连续
3. **双向上下文**：与自回归模型不同，能看到完整上下文

## 配置

### 配置文件：`configs/bert_config.yaml`

```yaml
model_name: "bert"
vocab_size: 30522              # 词汇表大小
hidden_size: 768               # 隐藏维度
num_hidden_layers: 12          # Transformer层数
num_attention_heads: 12        # 注意力头数
intermediate_size: 3072        # 前馈网络大小
max_position_embeddings: 512   # 最大序列长度
type_vocab_size: 2            # 标记类型数量（段落）
attention_probs_dropout_prob: 0.1  # 注意力dropout
hidden_dropout_prob: 0.1      # 隐藏层dropout
layer_norm_eps: 1e-12         # 层归一化epsilon
```

### 关键参数

- **`hidden_size`**：模型维度（base为768，large为1024）
- **`num_hidden_layers`**：模型深度（base为12，large为24）
- **`num_attention_heads`**：每层的注意力头数
- **`max_position_embeddings`**：模型能处理的最大序列长度
- **`vocab_size`**：词汇表大小（WordPiece通常为30,522）

## 使用示例

### BERT基本用法

```python
import torch
from models import BERT
from utils import Config

# Load configuration
config = Config.from_file('configs/bert_config.yaml')

# Create model
model = BERT(config.to_dict())
model.eval()

# Prepare input
batch_size = 2
seq_length = 128
vocab_size = config.get('vocab_size')

input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
attention_mask = torch.ones(batch_size, seq_length)
token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)

# Forward pass
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )

print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
print(f"Pooler output shape: {outputs['pooler_output'].shape}")
```

### Masked Language Modeling

```python
from models import BertForMaskedLM
from utils import Config

# Load configuration
config = Config.from_file('configs/bert_config.yaml')

# Create MLM model
mlm_model = BertForMaskedLM(config.to_dict())
mlm_model.eval()

# Prepare masked input
batch_size = 2
seq_length = 128
vocab_size = config.get('vocab_size')

# Create input with some [MASK] tokens (token id = 103)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
attention_mask = torch.ones(batch_size, seq_length)

# Mask some tokens
mask_positions = torch.randint(0, seq_length, (batch_size, 5))
for i in range(batch_size):
    input_ids[i, mask_positions[i]] = 103  # [MASK] token

# Create labels (only compute loss on masked positions)
labels = input_ids.clone()
labels[labels != 103] = -100  # Ignore non-masked tokens in loss

# Forward pass
with torch.no_grad():
    outputs = mlm_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

print(f"MLM logits shape: {outputs['logits'].shape}")
print(f"MLM loss: {outputs['loss'].item():.6f}")

# Get predictions for masked tokens
predictions = torch.argmax(outputs['logits'], dim=-1)
print(f"Predicted tokens for masked positions: {predictions[0, mask_positions[0]]}")
```

### Sequence Classification

```python
from models import BertForSequenceClassification
from utils import Config

# Load configuration
config = Config.from_file('configs/bert_config.yaml')
config.set('num_labels', 2)  # Binary classification

# Create classification model
cls_model = BertForSequenceClassification(config.to_dict())
cls_model.eval()

# Prepare input
batch_size = 4
seq_length = 128
input_ids = torch.randint(0, config.get('vocab_size'), (batch_size, seq_length))
attention_mask = torch.ones(batch_size, seq_length)
labels = torch.randint(0, 2, (batch_size,))  # Binary labels

# Forward pass
with torch.no_grad():
    outputs = cls_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

print(f"Classification logits shape: {outputs['logits'].shape}")
print(f"Classification loss: {outputs['loss'].item():.6f}")

# Get predictions
predictions = torch.argmax(outputs['logits'], dim=-1)
print(f"Predictions: {predictions}")
print(f"Accuracy: {(predictions == labels).float().mean().item():.3f}")
```

## Training

### Masked Language Modeling Pre-training

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import BertForMaskedLM
from utils import Config

# Configuration
config = Config.from_file('configs/bert_config.yaml')
model = BertForMaskedLM(config.to_dict())

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=1000
)

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Mask tokens for MLM training."""
    labels = inputs.clone()
    
    # Create random array of floats with equal dimensions to input_ids
    probability_matrix = torch.full(labels.shape, mlm_probability)
    
    # Mask tokens
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens
    
    # 80% of the time, replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = 103  # [MASK] token
    
    # 10% of the time, replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(config.get('vocab_size'), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    
    # 10% of the time, keep original token (indices_replaced & indices_random are False)
    
    return inputs, labels

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Mask tokens
        masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
```

### Fine-tuning for Classification

```python
from models import BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

def train_classification(model, train_dataloader, val_dataloader, num_epochs=3):
    """Fine-tune BERT for sequence classification."""
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        accuracy = accuracy_score(val_labels, val_predictions)
        f1 = f1_score(val_labels, val_predictions, average='weighted')
        
        print(f'Epoch {epoch}:')
        print(f'  Train Loss: {total_loss/len(train_dataloader):.6f}')
        print(f'  Val Accuracy: {accuracy:.4f}')
        print(f'  Val F1: {f1:.4f}')
        
        model.train()

# Usage
config = Config.from_file('configs/bert_config.yaml')
config.set('num_labels', num_classes)
model = BertForSequenceClassification(config.to_dict())
model = model.to(device)

train_classification(model, train_dataloader, val_dataloader)
```

## Advanced Usage

### Attention Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(model, input_ids, attention_mask, layer_idx=-1, head_idx=0):
    """Visualize attention weights."""
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
    
    # Get attention weights from specified layer
    attention_weights = outputs['attentions'][layer_idx]  # [batch, heads, seq, seq]
    attention = attention_weights[0, head_idx].cpu().numpy()  # First sample, specified head
    
    # Plot attention matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(attention, cmap='Blues')
    plt.colorbar()
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()
    
    return attention

# Example usage
input_ids = torch.randint(0, vocab_size, (1, 20))  # Single sentence
attention_mask = torch.ones(1, 20)

attention_matrix = visualize_attention(model, input_ids, attention_mask)
```

### Custom BERT Variants

```python
class BertForTokenClassification(nn.Module):
    """BERT for token-level classification (NER, POS tagging)."""
    
    def __init__(self, config):
        super().__init__()
        self.bert = BERT(config)
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob', 0.1))
        self.classifier = nn.Linear(config['hidden_size'], config['num_labels'])
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs['last_hidden_state']
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # Flatten for computing loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fn.ignore_index).type_as(labels)
            )
            loss = loss_fn(active_logits, active_labels)
        
        return {
            'loss': loss,
            'logits': logits
        }

# Usage for NER
config = Config.from_file('configs/bert_config.yaml')
config.set('num_labels', 9)  # BIO tagging for 4 entity types
ner_model = BertForTokenClassification(config.to_dict())
```

### Multi-GPU Training

```python
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_training():
    """Setup for distributed training."""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        
        # DataParallel (simpler but less efficient)
        model = nn.DataParallel(model)
        
        # Or DistributedDataParallel (more efficient)
        # model = DDP(model, device_ids=[local_rank])
    
    return model

# Gradient accumulation for large batch sizes
def train_with_accumulation(model, dataloader, accumulation_steps=4):
    """Training with gradient accumulation."""
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        outputs = model(**batch)
        loss = outputs['loss'] / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

## Model Variants

### Different Model Sizes

```python
# BERT Base
bert_base_config = {
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072
}

# BERT Large
bert_large_config = {
    'hidden_size': 1024,
    'num_hidden_layers': 24,
    'num_attention_heads': 16,
    'intermediate_size': 4096
}

# BERT Small (for resource-constrained environments)
bert_small_config = {
    'hidden_size': 512,
    'num_hidden_layers': 4,
    'num_attention_heads': 8,
    'intermediate_size': 2048
}
```

### Task-Specific Configurations

```python
# For sequence classification
classification_config = bert_base_config.copy()
classification_config.update({
    'num_labels': 3,  # Number of classes
    'problem_type': 'single_label_classification'
})

# For question answering
qa_config = bert_base_config.copy()
qa_config.update({
    'num_labels': 2,  # Start and end positions
})

# For multiple choice
multiple_choice_config = bert_base_config.copy()
multiple_choice_config.update({
    'num_choices': 4,  # Number of choices per question
})
```

## Tips and Best Practices

### Training Tips

1. **Learning Rate**: Use smaller learning rates for fine-tuning (2e-5 to 5e-5)
2. **Warmup**: Use learning rate warmup for the first 10% of training
3. **Gradient Clipping**: Clip gradients to prevent exploding gradients
4. **Layer-wise Learning Rate Decay**: Lower learning rates for lower layers

### Data Preprocessing

```python
def preprocess_text_for_bert(texts, tokenizer, max_length=512):
    """Preprocess text data for BERT."""
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'token_type_ids': encoded.get('token_type_ids', None)
    }
```

### Performance Optimization

1. **Mixed Precision**: Use automatic mixed precision for faster training
2. **Dynamic Padding**: Pad sequences to the maximum length in each batch
3. **Efficient Attention**: Use FlashAttention for memory efficiency

### Common Issues

1. **Out of Memory**: Reduce batch size, use gradient accumulation, or longer sequences
2. **Slow Convergence**: Check learning rate, use warmup, verify data quality
3. **Poor Performance**: Ensure proper tokenization, check for data leakage

## Paper Reference

- **Title**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- **Authors**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **Conference**: NAACL 2019
- **ArXiv**: https://arxiv.org/abs/1810.04805
- **Code**: https://github.com/google-research/bert