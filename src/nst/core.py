"""Neural Style Transfer implementation with modern PyTorch."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_image(
    image_path: str, 
    max_size: int = 400,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Load and preprocess an image for neural style transfer.
    
    Args:
        image_path: Path to the image file.
        max_size: Maximum size for the image (maintains aspect ratio).
        device: Device to load the image on.
        
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    if device is None:
        device = get_device()
        
    image = Image.open(image_path).convert("RGB")
    
    # Scale the image to the maximum size while maintaining aspect ratio
    scale = max_size / float(max(image.size))
    new_size = tuple([int(dim * scale) for dim in image.size])
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Apply the necessary transformations to the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    return image


def content_loss(content: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute content loss between content and target features.
    
    Args:
        content: Content features from VGG.
        target: Target features from VGG.
        
    Returns:
        torch.Tensor: Content loss value.
    """
    return torch.mean((content - target) ** 2)


def style_loss(style: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute style loss using Gram matrix.
    
    Args:
        style: Style features from VGG.
        target: Target features from VGG.
        
    Returns:
        torch.Tensor: Style loss value.
    """
    # Compute the Gram matrix of the style image
    gram_style = torch.mm(
        style.view(style.size(1), -1), 
        style.view(style.size(1), -1).t()
    )
    gram_target = torch.mm(
        target.view(target.size(1), -1), 
        target.view(target.size(1), -1).t()
    )
    return torch.mean((gram_style - gram_target) ** 2)


class NSTModel(nn.Module):
    """Neural Style Transfer model using VGG19 feature extractor.
    
    This model extracts features from different layers of a pre-trained VGG19
    network for content and style transfer.
    """
    
    def __init__(self, pretrained: bool = True):
        """Initialize the NST model.
        
        Args:
            pretrained: Whether to use pre-trained VGG19 weights.
        """
        super().__init__()
        vgg = models.vgg19(pretrained=pretrained).features.eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg
        self.content_layers = [4]  # Layer to extract content
        self.style_layers = [1, 6, 11, 20]  # Layers to extract style
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through the model.
        
        Args:
            x: Input image tensor.
            
        Returns:
            Tuple of (content_features, style_features).
        """
        content_features = []
        style_features = []
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.content_layers:
                content_features.append(x)
            if i in self.style_layers:
                style_features.append(x)
                
        return content_features, style_features


class NeuralStyleTransfer:
    """Neural Style Transfer implementation with modern PyTorch."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        seed: int = 42,
        style_weight: float = 1e6,
        content_weight: float = 1.0,
    ):
        """Initialize the Neural Style Transfer system.
        
        Args:
            device: Device to run on.
            seed: Random seed for reproducibility.
            style_weight: Weight for style loss.
            content_weight: Weight for content loss.
        """
        set_seed(seed)
        self.device = device or get_device()
        self.style_weight = style_weight
        self.content_weight = content_weight
        
        self.model = NSTModel().to(self.device)
        self.content_image: Optional[torch.Tensor] = None
        self.style_image: Optional[torch.Tensor] = None
        self.target_image: Optional[torch.Tensor] = None
        
    def load_images(
        self, 
        content_path: str, 
        style_path: str, 
        max_size: int = 400
    ) -> None:
        """Load content and style images.
        
        Args:
            content_path: Path to content image.
            style_path: Path to style image.
            max_size: Maximum size for images.
        """
        self.content_image = load_image(content_path, max_size, self.device)
        self.style_image = load_image(style_path, max_size, self.device)
        
        # Initialize target image as a copy of content image
        self.target_image = self.content_image.clone().requires_grad_(True)
        
    def transfer_style(
        self, 
        num_epochs: int = 500,
        learning_rate: float = 1.0,
        optimizer_type: str = "LBFGS"
    ) -> torch.Tensor:
        """Perform neural style transfer.
        
        Args:
            num_epochs: Number of optimization epochs.
            learning_rate: Learning rate for optimizer.
            optimizer_type: Type of optimizer ("LBFGS" or "Adam").
            
        Returns:
            torch.Tensor: Stylized image.
        """
        if self.content_image is None or self.style_image is None:
            raise ValueError("Images must be loaded first using load_images()")
            
        # Extract features from content and style images
        content_features, _ = self.model(self.content_image)
        _, style_features = self.model(self.style_image)
        
        # Set up optimizer
        if optimizer_type == "LBFGS":
            optimizer = optim.LBFGS([self.target_image], lr=learning_rate)
        else:
            optimizer = optim.Adam([self.target_image], lr=learning_rate)
            
        # Training loop
        for epoch in range(num_epochs):
            def closure():
                # Clamp values to valid range
                self.target_image.data.clamp_(0, 1)
                optimizer.zero_grad()
                
                # Extract features from target image
                target_content_features, target_style_features = self.model(self.target_image)
                
                # Compute losses
                content_loss_value = content_loss(
                    content_features[0], target_content_features[0]
                )
                
                style_loss_value = 0.0
                for style_feat, target_feat in zip(style_features, target_style_features):
                    style_loss_value += style_loss(style_feat, target_feat)
                
                # Combine losses
                total_loss = (
                    self.content_weight * content_loss_value + 
                    self.style_weight * style_loss_value
                )
                
                total_loss.backward()
                return total_loss
                
            optimizer.step(closure)
            
            # Display progress
            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    current_loss = closure()
                    print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {current_loss.item():.4f}")
                    
        return self.target_image.detach()
        
    def save_result(self, output_path: str) -> None:
        """Save the stylized image.
        
        Args:
            output_path: Path to save the output image.
        """
        if self.target_image is None:
            raise ValueError("No stylized image available. Run transfer_style() first.")
            
        # Denormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        denormalized = self.target_image * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        # Convert to PIL and save
        image = transforms.ToPILImage()(denormalized.squeeze(0).cpu())
        image.save(output_path)


def main():
    """Example usage of Neural Style Transfer."""
    # Set up the NST system
    nst = NeuralStyleTransfer(seed=42)
    
    # Load images (you'll need to provide actual image paths)
    try:
        nst.load_images(
            content_path="data/content.jpg",
            style_path="data/style.jpg",
            max_size=400
        )
        
        # Perform style transfer
        result = nst.transfer_style(num_epochs=500)
        
        # Save result
        nst.save_result("assets/stylized_output.jpg")
        print("Style transfer completed! Check assets/stylized_output.jpg")
        
    except FileNotFoundError as e:
        print(f"Image file not found: {e}")
        print("Please provide valid image paths in the load_images() call")


if __name__ == "__main__":
    main()
