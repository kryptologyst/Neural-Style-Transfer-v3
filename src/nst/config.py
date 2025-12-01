"""Configuration management for Neural Style Transfer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from omegaconf import OmegaConf, DictConfig
import yaml


class Config:
    """Configuration manager for Neural Style Transfer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _load_default_config(self) -> DictConfig:
        """Load default configuration.
        
        Returns:
            Default configuration.
        """
        default_config = {
            # Model settings
            'model': {
                'type': 'vgg19',  # 'vgg19', 'fast_nst', 'multi_scale'
                'pretrained': True,
                'content_layers': [4],
                'style_layers': [1, 6, 11, 20],
                'max_size': 512,
            },
            
            # Training settings
            'training': {
                'num_epochs': 500,
                'learning_rate': 1.0,
                'optimizer': 'LBFGS',  # 'LBFGS', 'Adam'
                'style_weight': 1e6,
                'content_weight': 1.0,
                'tv_weight': 1e-4,
                'batch_size': 1,
                'num_workers': 4,
            },
            
            # Data settings
            'data': {
                'content_dir': 'data/content',
                'style_dir': 'data/style',
                'output_dir': 'assets',
                'paired': False,
                'augmentation': {
                    'horizontal_flip': 0.5,
                    'color_jitter': {
                        'brightness': 0.1,
                        'contrast': 0.1,
                        'saturation': 0.1,
                        'hue': 0.1
                    }
                }
            },
            
            # Evaluation settings
            'evaluation': {
                'metrics': ['content_preservation', 'style_similarity', 'perceptual_distance'],
                'save_samples': True,
                'sample_frequency': 50,
                'num_eval_samples': 10
            },
            
            # System settings
            'system': {
                'device': 'auto',  # 'auto', 'cuda', 'mps', 'cpu'
                'seed': 42,
                'mixed_precision': False,
                'compile_model': False,
            },
            
            # Logging settings
            'logging': {
                'log_level': 'INFO',
                'log_file': 'logs/nst.log',
                'wandb': {
                    'enabled': False,
                    'project': 'neural-style-transfer',
                    'entity': None
                },
                'tensorboard': {
                    'enabled': False,
                    'log_dir': 'runs'
                }
            }
        }
        
        return OmegaConf.create(default_config)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file.
        """
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = OmegaConf.load(config_path)
        else:
            with open(config_path, 'r') as f:
                config = OmegaConf.create(yaml.safe_load(f))
        
        self.config = OmegaConf.merge(self.config, config)
        self.config_path = config_path
    
    def save_config(self, output_path: str) -> None:
        """Save configuration to file.
        
        Args:
            output_path: Path to save configuration.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            OmegaConf.save(self.config, output_path)
        else:
            with open(output_path, 'w') as f:
                yaml.dump(OmegaConf.to_container(self.config), f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self.config, key, default=default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        OmegaConf.set(self.config, key, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates.
        """
        self.config = OmegaConf.merge(self.config, updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary.
        """
        return OmegaConf.to_container(self.config)
    
    def validate(self) -> List[str]:
        """Validate configuration.
        
        Returns:
            List of validation errors.
        """
        errors = []
        
        # Check required paths
        if not os.path.exists(self.get('data.content_dir')):
            errors.append(f"Content directory does not exist: {self.get('data.content_dir')}")
        
        if not os.path.exists(self.get('data.style_dir')):
            errors.append(f"Style directory does not exist: {self.get('data.style_dir')}")
        
        # Check model type
        valid_models = ['vgg19', 'fast_nst', 'multi_scale']
        if self.get('model.type') not in valid_models:
            errors.append(f"Invalid model type: {self.get('model.type')}. Must be one of {valid_models}")
        
        # Check optimizer
        valid_optimizers = ['LBFGS', 'Adam']
        if self.get('training.optimizer') not in valid_optimizers:
            errors.append(f"Invalid optimizer: {self.get('training.optimizer')}. Must be one of {valid_optimizers}")
        
        # Check device
        valid_devices = ['auto', 'cuda', 'mps', 'cpu']
        if self.get('system.device') not in valid_devices:
            errors.append(f"Invalid device: {self.get('system.device')}. Must be one of {valid_devices}")
        
        return errors


def create_config_from_args(args: Dict[str, Any]) -> Config:
    """Create configuration from command line arguments.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Configuration object.
    """
    config = Config()
    
    # Update configuration with command line arguments
    for key, value in args.items():
        if value is not None:
            config.set(key, value)
    
    return config


def load_config_from_file(config_path: str) -> Config:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration object.
    """
    return Config(config_path)
