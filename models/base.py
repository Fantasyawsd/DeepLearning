"""
Base model class for all deep learning models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


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
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_parameters(self, module_names: Optional[list] = None):
        """Freeze parameters of specified modules or all parameters if None."""
        if module_names is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, module in self.named_modules():
                if any(mod_name in name for mod_name in module_names):
                    for param in module.parameters():
                        param.requires_grad = False
    
    def unfreeze_parameters(self, module_names: Optional[list] = None):
        """Unfreeze parameters of specified modules or all parameters if None."""
        if module_names is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, module in self.named_modules():
                if any(mod_name in name for mod_name in module_names):
                    for param in module.parameters():
                        param.requires_grad = True
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'model_name': self.model_name,
        }
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, map_location: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint
    
    def summary(self):
        """Print model summary."""
        print(f"Model: {self.model_name}")
        print(f"Total parameters: {self.get_num_params():,}")
        print(f"Trainable parameters: {self.get_num_trainable_params():,}")
        print(f"Model size: {self.get_num_params() * 4 / 1024 / 1024:.2f} MB")