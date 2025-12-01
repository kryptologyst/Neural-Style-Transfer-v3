"""Evaluation metrics and utilities for Neural Style Transfer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from torchvision import models, transforms
from PIL import Image
import lpips
from clean_fid import fid


class StyleTransferEvaluator:
    """Evaluator for Neural Style Transfer models."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize evaluator.
        
        Args:
            device: Device to run evaluation on.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load VGG for feature extraction
        self.vgg = models.vgg19(pretrained=True).features.eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # Load LPIPS model
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        
    def content_preservation_score(
        self, 
        original: torch.Tensor, 
        stylized: torch.Tensor
    ) -> float:
        """Compute content preservation score using VGG features.
        
        Args:
            original: Original content image.
            stylized: Stylized image.
            
        Returns:
            Content preservation score (lower is better).
        """
        # Extract features from conv4_2 layer
        original_features = self._extract_features(original, layer_idx=26)
        stylized_features = self._extract_features(stylized, layer_idx=26)
        
        # Compute MSE loss
        mse_loss = F.mse_loss(original_features, stylized_features)
        return mse_loss.item()
    
    def style_similarity_score(
        self, 
        style_reference: torch.Tensor, 
        stylized: torch.Tensor
    ) -> float:
        """Compute style similarity score using Gram matrices.
        
        Args:
            style_reference: Reference style image.
            stylized: Stylized image.
            
        Returns:
            Style similarity score (lower is better).
        """
        style_layers = [3, 8, 17, 26, 35]  # relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
        total_loss = 0.0
        
        for layer_idx in style_layers:
            style_features = self._extract_features(style_reference, layer_idx)
            stylized_features = self._extract_features(stylized, layer_idx)
            
            # Compute Gram matrices
            style_gram = self._gram_matrix(style_features)
            stylized_gram = self._gram_matrix(stylized_features)
            
            # Compute MSE loss
            layer_loss = F.mse_loss(style_gram, stylized_gram)
            total_loss += layer_loss
            
        return total_loss.item()
    
    def perceptual_distance(
        self, 
        image1: torch.Tensor, 
        image2: torch.Tensor
    ) -> float:
        """Compute perceptual distance using LPIPS.
        
        Args:
            image1: First image.
            image2: Second image.
            
        Returns:
            Perceptual distance.
        """
        # Ensure images are in [-1, 1] range for LPIPS
        image1 = (image1 - 0.5) * 2
        image2 = (image2 - 0.5) * 2
        
        with torch.no_grad():
            distance = self.lpips_model(image1, image2)
        return distance.item()
    
    def _extract_features(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Extract features from VGG at specified layer.
        
        Args:
            x: Input tensor.
            layer_idx: Layer index.
            
        Returns:
            Extracted features.
        """
        features = x
        for i, layer in enumerate(self.vgg):
            features = layer(features)
            if i == layer_idx:
                break
        return features
    
    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix.
        
        Args:
            x: Input tensor.
            
        Returns:
            Gram matrix.
        """
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width)
        gram = torch.bmm(x, x.transpose(1, 2))
        return gram / (channels * height * width)
    
    def evaluate_batch(
        self, 
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        stylized_images: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate a batch of stylized images.
        
        Args:
            content_images: Original content images.
            style_images: Reference style images.
            stylized_images: Generated stylized images.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        batch_size = content_images.size(0)
        
        content_scores = []
        style_scores = []
        perceptual_scores = []
        
        for i in range(batch_size):
            # Content preservation
            content_score = self.content_preservation_score(
                content_images[i:i+1], stylized_images[i:i+1]
            )
            content_scores.append(content_score)
            
            # Style similarity
            style_score = self.style_similarity_score(
                style_images[i:i+1], stylized_images[i:i+1]
            )
            style_scores.append(style_score)
            
            # Perceptual distance from content
            perceptual_score = self.perceptual_distance(
                content_images[i:i+1], stylized_images[i:i+1]
            )
            perceptual_scores.append(perceptual_score)
        
        return {
            'content_preservation': np.mean(content_scores),
            'style_similarity': np.mean(style_scores),
            'perceptual_distance': np.mean(perceptual_scores),
            'content_preservation_std': np.std(content_scores),
            'style_similarity_std': np.std(style_scores),
            'perceptual_distance_std': np.std(perceptual_scores)
        }


class FIDCalculator:
    """FID calculator for style transfer evaluation."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize FID calculator.
        
        Args:
            device: Device to run on.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def calculate_fid(
        self, 
        real_images: List[str], 
        generated_images: List[str]
    ) -> float:
        """Calculate FID between real and generated images.
        
        Args:
            real_images: List of paths to real images.
            generated_images: List of paths to generated images.
            
        Returns:
            FID score.
        """
        try:
            fid_score = fid.compute_fid(
                real_images, 
                generated_images, 
                device=self.device,
                batch_size=50
            )
            return fid_score
        except Exception as e:
            print(f"Error calculating FID: {e}")
            return float('inf')


def create_evaluation_report(
    evaluator: StyleTransferEvaluator,
    content_images: torch.Tensor,
    style_images: torch.Tensor,
    stylized_images: torch.Tensor,
    model_name: str = "NST Model"
) -> Dict[str, Any]:
    """Create comprehensive evaluation report.
    
    Args:
        evaluator: Style transfer evaluator.
        content_images: Original content images.
        style_images: Reference style images.
        stylized_images: Generated stylized images.
        model_name: Name of the model being evaluated.
        
    Returns:
        Evaluation report dictionary.
    """
    # Compute metrics
    metrics = evaluator.evaluate_batch(content_images, style_images, stylized_images)
    
    # Create report
    report = {
        'model_name': model_name,
        'num_samples': content_images.size(0),
        'metrics': metrics,
        'summary': {
            'content_preservation_quality': 'Good' if metrics['content_preservation'] < 0.1 else 'Fair' if metrics['content_preservation'] < 0.3 else 'Poor',
            'style_transfer_quality': 'Good' if metrics['style_similarity'] < 0.1 else 'Fair' if metrics['style_similarity'] < 0.3 else 'Poor',
            'overall_perceptual_quality': 'Good' if metrics['perceptual_distance'] < 0.3 else 'Fair' if metrics['perceptual_distance'] < 0.5 else 'Poor'
        }
    }
    
    return report


def print_evaluation_report(report: Dict[str, Any]) -> None:
    """Print evaluation report in a formatted way.
    
    Args:
        report: Evaluation report dictionary.
    """
    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT: {report['model_name']}")
    print(f"{'='*50}")
    print(f"Number of samples: {report['num_samples']}")
    print(f"\nMETRICS:")
    print(f"  Content Preservation: {report['metrics']['content_preservation']:.4f} ± {report['metrics']['content_preservation_std']:.4f}")
    print(f"  Style Similarity: {report['metrics']['style_similarity']:.4f} ± {report['metrics']['style_similarity_std']:.4f}")
    print(f"  Perceptual Distance: {report['metrics']['perceptual_distance']:.4f} ± {report['metrics']['perceptual_distance_std']:.4f}")
    print(f"\nQUALITY ASSESSMENT:")
    print(f"  Content Preservation: {report['summary']['content_preservation_quality']}")
    print(f"  Style Transfer: {report['summary']['style_transfer_quality']}")
    print(f"  Overall Quality: {report['summary']['overall_perceptual_quality']}")
    print(f"{'='*50}\n")
