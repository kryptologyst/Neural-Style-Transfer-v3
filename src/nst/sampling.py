"""Sampling and inference utilities for Neural Style Transfer."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .core import NeuralStyleTransfer, set_seed, get_device
from .models import FastNSTModel, MultiScaleStyleTransfer
from .config import Config


class StyleTransferSampler:
    """Sampler for Neural Style Transfer inference."""
    
    def __init__(self, config: Config):
        """Initialize sampler.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.device = get_device() if config.get('system.device') == 'auto' else torch.device(config.get('system.device'))
        
        # Set seed for reproducibility
        set_seed(config.get('system.seed'))
        
        # Initialize model
        self.model = self._load_model()
        
    def _load_model(self) -> nn.Module:
        """Load the appropriate model based on configuration.
        
        Returns:
            Loaded model.
        """
        model_type = self.config.get('model.type')
        
        if model_type == 'vgg19':
            return NeuralStyleTransfer(
                device=self.device,
                seed=self.config.get('system.seed'),
                style_weight=self.config.get('training.style_weight'),
                content_weight=self.config.get('training.content_weight')
            )
        elif model_type == 'fast_nst':
            return FastNSTModel().to(self.device)
        elif model_type == 'multi_scale':
            scales = self.config.get('model.scales', [256, 512, 1024])
            return MultiScaleStyleTransfer(scales).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def load_images(
        self, 
        content_path: str, 
        style_path: str,
        max_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load content and style images.
        
        Args:
            content_path: Path to content image.
            style_path: Path to style image.
            max_size: Maximum image size (overrides config).
            
        Returns:
            Tuple of (content_tensor, style_tensor).
        """
        max_size = max_size or self.config.get('model.max_size')
        
        # Load images
        content_image = Image.open(content_path).convert('RGB')
        style_image = Image.open(style_path).convert('RGB')
        
        # Resize images
        content_image = self._resize_image(content_image, max_size)
        style_image = self._resize_image(style_image, max_size)
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        content_tensor = transform(content_image).unsqueeze(0).to(self.device)
        style_tensor = transform(style_image).unsqueeze(0).to(self.device)
        
        return content_tensor, style_tensor
    
    def _resize_image(self, image: Image.Image, max_size: int) -> Image.Image:
        """Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image.
            max_size: Maximum size.
            
        Returns:
            Resized image.
        """
        scale = max_size / float(max(image.size))
        new_size = tuple([int(dim * scale) for dim in image.size])
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    def transfer_style(
        self, 
        content_path: str, 
        style_path: str,
        output_path: Optional[str] = None,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> torch.Tensor:
        """Perform style transfer.
        
        Args:
            content_path: Path to content image.
            style_path: Path to style image.
            output_path: Path to save output image.
            num_epochs: Number of optimization epochs.
            learning_rate: Learning rate for optimization.
            
        Returns:
            Stylized image tensor.
        """
        # Load images
        content_tensor, style_tensor = self.load_images(content_path, style_path)
        
        # Get parameters
        num_epochs = num_epochs or self.config.get('training.num_epochs')
        learning_rate = learning_rate or self.config.get('training.learning_rate')
        
        # Perform style transfer based on model type
        model_type = self.config.get('model.type')
        
        if model_type == 'vgg19':
            # Use iterative optimization
            self.model.load_images(content_path, style_path)
            stylized = self.model.transfer_style(num_epochs, learning_rate)
            
            if output_path:
                self.model.save_result(output_path)
                
        elif model_type in ['fast_nst', 'multi_scale']:
            # Use feedforward model
            stylized = self.model(content_tensor)
            
            if output_path:
                self._save_tensor_image(stylized, output_path)
        
        return stylized
    
    def batch_transfer(
        self, 
        content_paths: List[str], 
        style_paths: List[str],
        output_dir: str,
        num_epochs: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Perform batch style transfer.
        
        Args:
            content_paths: List of content image paths.
            style_paths: List of style image paths.
            output_dir: Output directory.
            num_epochs: Number of optimization epochs.
            
        Returns:
            List of stylized image tensors.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, (content_path, style_path) in enumerate(tqdm(zip(content_paths, style_paths))):
            try:
                output_file = output_path / f"stylized_{i:03d}.jpg"
                stylized = self.transfer_style(
                    content_path, 
                    style_path, 
                    str(output_file),
                    num_epochs
                )
                results.append(stylized)
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
        
        return results
    
    def interpolate_styles(
        self, 
        content_path: str, 
        style_paths: List[str],
        output_dir: str,
        num_interpolations: int = 5
    ) -> List[torch.Tensor]:
        """Interpolate between multiple styles.
        
        Args:
            content_path: Path to content image.
            style_paths: List of style image paths.
            output_dir: Output directory.
            num_interpolations: Number of interpolation steps.
            
        Returns:
            List of interpolated stylized images.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load content image
        content_tensor, _ = self.load_images(content_path, style_paths[0])
        
        results = []
        
        for i in range(num_interpolations):
            # Select style images for interpolation
            if len(style_paths) == 1:
                style_path = style_paths[0]
            else:
                # Interpolate between first and second style
                alpha = i / (num_interpolations - 1)
                style_path = style_paths[0] if alpha < 0.5 else style_paths[1]
            
            # Perform style transfer
            output_file = output_path / f"interpolated_{i:03d}.jpg"
            stylized = self.transfer_style(content_path, style_path, str(output_file))
            results.append(stylized)
        
        return results
    
    def _save_tensor_image(self, tensor: torch.Tensor, output_path: str) -> None:
        """Save tensor as image.
        
        Args:
            tensor: Image tensor.
            output_path: Output path.
        """
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        denormalized = tensor * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        # Convert to PIL and save
        image = transforms.ToPILImage()(denormalized.squeeze(0).cpu())
        image.save(output_path)


def create_sample_grid(
    content_images: List[torch.Tensor],
    style_images: List[torch.Tensor], 
    stylized_images: List[torch.Tensor],
    output_path: str,
    max_images: int = 6
) -> None:
    """Create a sample grid showing content, style, and stylized images.
    
    Args:
        content_images: List of content image tensors.
        style_images: List of style image tensors.
        stylized_images: List of stylized image tensors.
        output_path: Path to save the grid.
        max_images: Maximum number of images to show.
    """
    num_images = min(len(content_images), max_images)
    
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 3, 9))
    if num_images == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_images):
        # Content image
        content = content_images[i].squeeze(0).cpu().permute(1, 2, 0).numpy()
        content = np.clip(content, 0, 1)
        axes[0, i].imshow(content)
        axes[0, i].set_title(f'Content {i+1}')
        axes[0, i].axis('off')
        
        # Style image
        style = style_images[i].squeeze(0).cpu().permute(1, 2, 0).numpy()
        style = np.clip(style, 0, 1)
        axes[1, i].imshow(style)
        axes[1, i].set_title(f'Style {i+1}')
        axes[1, i].axis('off')
        
        # Stylized image
        stylized = stylized_images[i].squeeze(0).cpu().permute(1, 2, 0).numpy()
        stylized = np.clip(stylized, 0, 1)
        axes[2, i].imshow(stylized)
        axes[2, i].set_title(f'Stylized {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def random_style_transfer(
    content_dir: str, 
    style_dir: str, 
    output_dir: str,
    config: Config,
    num_samples: int = 5
) -> None:
    """Perform random style transfer on sample images.
    
    Args:
        content_dir: Directory containing content images.
        style_dir: Directory containing style images.
        output_dir: Output directory.
        config: Configuration object.
        num_samples: Number of random samples to generate.
    """
    sampler = StyleTransferSampler(config)
    
    # Get random image pairs
    content_files = list(Path(content_dir).glob('*.jpg')) + list(Path(content_dir).glob('*.png'))
    style_files = list(Path(style_dir).glob('*.jpg')) + list(Path(style_dir).glob('*.png'))
    
    if not content_files or not style_files:
        print("No images found in content or style directories")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        content_path = random.choice(content_files)
        style_path = random.choice(style_files)
        
        output_file = output_path / f"random_sample_{i:03d}.jpg"
        
        try:
            sampler.transfer_style(str(content_path), str(style_path), str(output_file))
            print(f"Generated sample {i+1}/{num_samples}: {output_file}")
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
