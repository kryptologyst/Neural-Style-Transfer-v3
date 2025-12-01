"""Advanced Neural Style Transfer models and techniques."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import math


class AdaIN(nn.Module):
    """Adaptive Instance Normalization for style transfer."""
    
    def __init__(self, eps: float = 1e-5):
        """Initialize AdaIN layer.
        
        Args:
            eps: Small value for numerical stability.
        """
        super().__init__()
        self.eps = eps
        
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply adaptive instance normalization.
        
        Args:
            content: Content features.
            style: Style features.
            
        Returns:
            torch.Tensor: Normalized features.
        """
        # Calculate mean and variance for content and style
        content_mean = torch.mean(content, dim=[2, 3], keepdim=True)
        content_var = torch.var(content, dim=[2, 3], keepdim=True)
        
        style_mean = torch.mean(style, dim=[2, 3], keepdim=True)
        style_var = torch.var(style, dim=[2, 3], keepdim=True)
        
        # Normalize content and apply style statistics
        content_norm = (content - content_mean) / torch.sqrt(content_var + self.eps)
        return style_var * content_norm + style_mean


class FastNSTModel(nn.Module):
    """Fast Neural Style Transfer model using encoder-decoder architecture."""
    
    def __init__(self, input_channels: int = 3, output_channels: int = 3):
        """Initialize Fast NST model.
        
        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(5)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 9, padding=4),
            nn.Tanh(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input image tensor.
            
        Returns:
            torch.Tensor: Stylized image.
        """
        # Encode
        x = self.encoder(x)
        
        # Apply residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
            
        # Decode
        x = self.decoder(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for Fast NST."""
    
    def __init__(self, channels: int):
        """Initialize residual block.
        
        Args:
            channels: Number of channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.
        
        Args:
            x: Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class StyleEncoder(nn.Module):
    """Style encoder for extracting style features."""
    
    def __init__(self):
        """Initialize style encoder."""
        super().__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.features = vgg.features[:36]  # Up to conv5_4
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract style features.
        
        Args:
            x: Input image tensor.
            
        Returns:
            List[torch.Tensor]: Style features from different layers.
        """
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [3, 8, 17, 26, 35]:  # relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
                features.append(x)
        return features


class ContentEncoder(nn.Module):
    """Content encoder for extracting content features."""
    
    def __init__(self):
        """Initialize content encoder."""
        super().__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.features = vgg.features[:36]  # Up to conv5_4
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract content features.
        
        Args:
            x: Input image tensor.
            
        Returns:
            torch.Tensor: Content features from conv4_2.
        """
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 26:  # conv4_2
                return x
        return x


class MultiScaleStyleTransfer(nn.Module):
    """Multi-scale style transfer model."""
    
    def __init__(self, scales: List[int] = [256, 512, 1024]):
        """Initialize multi-scale model.
        
        Args:
            scales: List of scales to process.
        """
        super().__init__()
        self.scales = scales
        self.models = nn.ModuleList([
            FastNSTModel() for _ in scales
        ])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass at multiple scales.
        
        Args:
            x: Input image tensor.
            
        Returns:
            List[torch.Tensor]: Stylized images at different scales.
        """
        results = []
        for i, scale in enumerate(self.scales):
            # Resize input to current scale
            resized = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
            # Apply style transfer
            stylized = self.models[i](resized)
            # Resize back to original size
            stylized = F.interpolate(stylized, size=x.shape[2:], mode='bilinear', align_corners=False)
            results.append(stylized)
        return results


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, feature_layers: List[int] = [3, 8, 17, 26, 35]):
        """Initialize perceptual loss.
        
        Args:
            feature_layers: VGG layers to use for loss computation.
        """
        super().__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.features = vgg.features
        self.feature_layers = feature_layers
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.
        
        Args:
            x: Input tensor.
            target: Target tensor.
            
        Returns:
            torch.Tensor: Perceptual loss.
        """
        loss = 0.0
        for i, layer in enumerate(self.features):
            x = layer(x)
            target = layer(target)
            if i in self.feature_layers:
                loss += F.mse_loss(x, target)
        return loss


class TotalVariationLoss(nn.Module):
    """Total variation loss for smoothness regularization."""
    
    def __init__(self):
        """Initialize total variation loss."""
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss.
        
        Args:
            x: Input tensor.
            
        Returns:
            torch.Tensor: Total variation loss.
        """
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        
    def _tensor_size(self, t: torch.Tensor) -> int:
        """Get tensor size."""
        return t.size()[1] * t.size()[2] * t.size()[3]
