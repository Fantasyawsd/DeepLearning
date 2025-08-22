"""
Shared base model class for all deep learning models.

This module provides the base class that can be imported from anywhere in the project.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all deep learning models.
    
    This class provides common functionality and interface that all models should implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = config.get('model_name', 'BaseModel')
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    def get_num_params(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> None:
        """Print model summary."""
        print(f"Model: {self.model_name}")
        print(f"Total parameters: {self.get_num_params():,}")
        print(f"Model structure:")
        print(self)
    
    def save_checkpoint(self, path: str, epoch: int = 0, optimizer_state: Optional[Dict] = None,
                       scheduler_state: Optional[Dict] = None, **kwargs) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_name': self.model_name,
            **kwargs
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str, map_location: str = 'cpu') -> Dict:
        """Load model checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Checkpoint loaded from: {path}")
        return checkpoint
    
    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze model parameters."""
        for param in self.parameters():
            param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        print(f"Model parameters {status}")
    
    def freeze_layers(self, layer_names: list) -> None:
        """Freeze specific layers by name."""
        for name, module in self.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                for param in module.parameters():
                    param.requires_grad = False
                print(f"Frozen layer: {name}")
    
    def get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def count_parameters_by_layer(self) -> Dict[str, int]:
        """Count parameters for each layer."""
        param_count = {}
        for name, module in self.named_modules():
            if list(module.children()) == []:  # Leaf modules only
                param_count[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return param_count
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb