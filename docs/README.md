# Documentation Index

Welcome to the Deep Learning Models Framework documentation! This section provides comprehensive guides and tutorials for using the framework and its models.

## 📚 Documentation Structure

### Main Guides
- **[User Guide](USER_GUIDE.md)** - Complete tutorial for getting started with the framework
- **[Getting Started Tutorial](TUTORIAL.md)** - Step-by-step tutorial with code examples
- **[Installation Guide](#installation)** - Step-by-step installation instructions
- **[Training Guide](#training)** - How to train models using the framework

### Model-Specific Guides
- **[MAE Guide](models/MAE_GUIDE.md)** - Complete guide for Masked Autoencoder
- **[BERT Guide](models/BERT_GUIDE.md)** - Complete guide for BERT and its variants
- **[Swin Transformer Guide](models/SWIN_TRANSFORMER_GUIDE.md)** - Complete guide for Swin Transformer

## 🚀 Quick Start

1. **Installation**: Follow the [User Guide](USER_GUIDE.md#getting-started) for installation
2. **Basic Usage**: Try the examples in the [examples/](../examples/) directory
3. **Model Training**: Use the training script or implement custom training loops
4. **Advanced Usage**: Explore model-specific guides for advanced features

## 📖 What Each Guide Contains

### User Guide
- Installation and setup
- Project structure overview
- Basic usage patterns
- Training instructions
- Advanced features
- Troubleshooting

### Model Guides
Each model guide includes:
- Model overview and key concepts
- Configuration options
- Usage examples (basic to advanced)
- Training instructions
- Visualization techniques
- Performance optimization tips
- Common issues and solutions

## 🔧 Configuration Management

All models use YAML configuration files located in [`configs/`](../configs/):
- `mae_config.yaml` - MAE configuration
- `bert_config.yaml` - BERT configuration  
- `swin_config.yaml` - Swin Transformer configuration

## 💡 Examples and Tutorials

Check out the [`examples/`](../examples/) directory for working code:
- `mae_example.py` - MAE usage example
- `bert_example.py` - BERT usage example
- `swin_transformer_example.py` - Swin Transformer usage example

## 🛠️ Development and Contributing

### Project Structure
```
docs/
├── README.md                    # This file
├── USER_GUIDE.md               # Main user guide
└── models/                     # Model-specific guides
    ├── MAE_GUIDE.md           # MAE documentation
    ├── BERT_GUIDE.md          # BERT documentation
    └── SWIN_TRANSFORMER_GUIDE.md # Swin Transformer documentation
```

### Adding New Documentation
1. Create new guide in appropriate directory
2. Follow existing format and structure
3. Include practical examples and code snippets
4. Update this index to reference the new guide

## 📞 Getting Help

1. **Check the relevant guide** for your model or use case
2. **Review the examples** in the `examples/` directory
3. **Check common issues** in the troubleshooting sections
4. **Open an issue** on GitHub if you need additional help

## 🔗 External Resources

### Papers
- **MAE**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- **Swin Transformer**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

### Official Implementations
- **MAE**: [Facebook Research MAE](https://github.com/facebookresearch/mae)
- **BERT**: [Google Research BERT](https://github.com/google-research/bert)
- **Swin Transformer**: [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer)

---

*This documentation is continuously updated. Last updated: 2024*