"""
Configuration management utilities.
"""

import json
import yaml
from typing import Dict, Any, Union
from pathlib import Path


class Config:
    """Configuration class for managing model and training configurations."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config = config_dict or {}
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def update(self, other: Union[Dict[str, Any], 'Config']):
        if isinstance(other, Config):
            self._config.update(other._config)
        else:
            self._config.update(other)
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Config':
        """Load configuration from JSON or YAML file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")
        
        return cls(config_dict)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to JSON or YAML file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            elif file_path.suffix.lower() == '.json':
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def __str__(self) -> str:
        return json.dumps(self._config, indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        return f"Config({self._config})"