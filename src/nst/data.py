"""Data pipeline and dataset utilities for Neural Style Transfer."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import requests
from tqdm import tqdm


class StyleTransferDataset(Dataset):
    """Dataset for Neural Style Transfer training."""
    
    def __init__(
        self,
        content_dir: str,
        style_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_size: int = 512,
        paired: bool = False
    ):
        """Initialize the dataset.
        
        Args:
            content_dir: Directory containing content images.
            style_dir: Directory containing style images.
            transform: Optional transforms to apply.
            max_size: Maximum image size.
            paired: Whether content and style images are paired.
        """
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.max_size = max_size
        self.paired = paired
        
        # Get image files
        self.content_files = self._get_image_files(self.content_dir)
        self.style_files = self._get_image_files(self.style_dir)
        
        if self.paired:
            assert len(self.content_files) == len(self.style_files), \
                "Paired dataset requires equal number of content and style images"
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((max_size, max_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files from directory.
        
        Args:
            directory: Directory to search.
            
        Returns:
            List of image file paths.
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        files = []
        for ext in image_extensions:
            files.extend(directory.glob(f'*{ext}'))
            files.extend(directory.glob(f'*{ext.upper()}'))
        return sorted(files)
    
    def __len__(self) -> int:
        """Get dataset length."""
        if self.paired:
            return len(self.content_files)
        return max(len(self.content_files), len(self.style_files))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset.
        
        Args:
            idx: Index.
            
        Returns:
            Dictionary containing content and style images.
        """
        # Get content image
        content_idx = idx % len(self.content_files)
        content_path = self.content_files[content_idx]
        content_image = Image.open(content_path).convert('RGB')
        
        # Get style image
        if self.paired:
            style_idx = idx % len(self.style_files)
        else:
            style_idx = random.randint(0, len(self.style_files) - 1)
        style_path = self.style_files[style_idx]
        style_image = Image.open(style_path).convert('RGB')
        
        # Apply transforms
        content_tensor = self.transform(content_image)
        style_tensor = self.transform(style_image)
        
        return {
            'content': content_tensor,
            'style': style_tensor,
            'content_path': str(content_path),
            'style_path': str(style_path)
        }


class SampleDatasetGenerator:
    """Generate sample datasets for testing and demonstration."""
    
    @staticmethod
    def create_sample_content_images(output_dir: str, num_images: int = 10) -> None:
        """Create sample content images.
        
        Args:
            output_dir: Output directory.
            num_images: Number of images to create.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download sample images from Unsplash
        sample_urls = [
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1501594907352-04dda38d4700?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1501594907352-04dda38d4700?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=512&h=512&fit=crop",
        ]
        
        for i in range(min(num_images, len(sample_urls))):
            try:
                response = requests.get(sample_urls[i])
                if response.status_code == 200:
                    image_path = output_path / f"content_{i:03d}.jpg"
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded content image {i+1}/{num_images}")
            except Exception as e:
                print(f"Failed to download image {i+1}: {e}")
    
    @staticmethod
    def create_sample_style_images(output_dir: str, num_images: int = 10) -> None:
        """Create sample style images.
        
        Args:
            output_dir: Output directory.
            num_images: Number of images to create.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download sample artistic images
        style_urls = [
            "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=512&h=512&fit=crop",
        ]
        
        for i in range(min(num_images, len(style_urls))):
            try:
                response = requests.get(style_urls[i])
                if response.status_code == 200:
                    image_path = output_path / f"style_{i:03d}.jpg"
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded style image {i+1}/{num_images}")
            except Exception as e:
                print(f"Failed to download image {i+1}: {e}")


def create_data_loaders(
    content_dir: str,
    style_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    max_size: int = 512,
    paired: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        content_dir: Directory containing content images.
        style_dir: Directory containing style images.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        max_size: Maximum image size.
        paired: Whether content and style images are paired.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = StyleTransferDataset(
        content_dir, style_dir, train_transform, max_size, paired
    )
    val_dataset = StyleTransferDataset(
        content_dir, style_dir, val_transform, max_size, paired
    )
    test_dataset = StyleTransferDataset(
        content_dir, style_dir, val_transform, max_size, paired
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize a tensor using ImageNet normalization.
    
    Args:
        tensor: Normalized tensor.
        
    Returns:
        Denormalized tensor.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean
