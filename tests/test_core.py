"""Unit tests for Neural Style Transfer core functionality."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nst import (
    NeuralStyleTransfer,
    set_seed,
    get_device,
    load_image,
    content_loss,
    style_loss,
    NSTModel
)


class TestCore:
    """Test core functionality."""
    
    def test_set_seed(self):
        """Test seed setting functionality."""
        set_seed(42)
        torch_rand1 = torch.rand(1)
        
        set_seed(42)
        torch_rand2 = torch.rand(1)
        
        assert torch.allclose(torch_rand1, torch_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_content_loss(self):
        """Test content loss computation."""
        # Create dummy tensors
        content = torch.randn(1, 64, 32, 32)
        target = torch.randn(1, 64, 32, 32)
        
        loss = content_loss(content, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])
    
    def test_style_loss(self):
        """Test style loss computation."""
        # Create dummy tensors
        style = torch.randn(1, 64, 32, 32)
        target = torch.randn(1, 64, 32, 32)
        
        loss = style_loss(style, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])
    
    def test_nst_model_initialization(self):
        """Test NST model initialization."""
        model = NSTModel()
        
        assert isinstance(model, NSTModel)
        assert hasattr(model, 'vgg')
        assert hasattr(model, 'content_layers')
        assert hasattr(model, 'style_layers')
    
    def test_nst_model_forward(self):
        """Test NST model forward pass."""
        model = NSTModel()
        
        # Create dummy input
        x = torch.randn(1, 3, 224, 224)
        
        content_features, style_features = model(x)
        
        assert isinstance(content_features, list)
        assert isinstance(style_features, list)
        assert len(content_features) == len(model.content_layers)
        assert len(style_features) == len(model.style_layers)
    
    def test_neural_style_transfer_initialization(self):
        """Test NeuralStyleTransfer initialization."""
        nst = NeuralStyleTransfer()
        
        assert isinstance(nst, NeuralStyleTransfer)
        assert hasattr(nst, 'model')
        assert hasattr(nst, 'device')
        assert hasattr(nst, 'style_weight')
        assert hasattr(nst, 'content_weight')
    
    def test_neural_style_transfer_device_assignment(self):
        """Test device assignment in NeuralStyleTransfer."""
        device = torch.device('cpu')
        nst = NeuralStyleTransfer(device=device)
        
        assert nst.device == device
        assert next(nst.model.parameters()).device == device


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_image_nonexistent(self):
        """Test loading non-existent image."""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_image.jpg")
    
    def test_load_image_invalid_path(self):
        """Test loading image with invalid path."""
        with pytest.raises(FileNotFoundError):
            load_image("")


class TestLossFunctions:
    """Test loss function computations."""
    
    def test_content_loss_identical_inputs(self):
        """Test content loss with identical inputs."""
        tensor = torch.randn(1, 64, 32, 32)
        loss = content_loss(tensor, tensor)
        
        assert torch.allclose(loss, torch.tensor(0.0))
    
    def test_style_loss_identical_inputs(self):
        """Test style loss with identical inputs."""
        tensor = torch.randn(1, 64, 32, 32)
        loss = style_loss(tensor, tensor)
        
        assert torch.allclose(loss, torch.tensor(0.0))
    
    def test_content_loss_different_inputs(self):
        """Test content loss with different inputs."""
        tensor1 = torch.randn(1, 64, 32, 32)
        tensor2 = torch.randn(1, 64, 32, 32)
        loss = content_loss(tensor1, tensor2)
        
        assert loss.item() > 0
    
    def test_style_loss_different_inputs(self):
        """Test style loss with different inputs."""
        tensor1 = torch.randn(1, 64, 32, 32)
        tensor2 = torch.randn(1, 64, 32, 32)
        loss = style_loss(tensor1, tensor2)
        
        assert loss.item() > 0


class TestModelArchitecture:
    """Test model architecture components."""
    
    def test_vgg_frozen_parameters(self):
        """Test that VGG parameters are frozen."""
        model = NSTModel()
        
        for param in model.vgg.parameters():
            assert not param.requires_grad
    
    def test_model_output_shapes(self):
        """Test model output shapes."""
        model = NSTModel()
        x = torch.randn(1, 3, 224, 224)
        
        content_features, style_features = model(x)
        
        # Check that features have reasonable shapes
        for feature in content_features:
            assert feature.dim() == 4  # [batch, channels, height, width]
            assert feature.size(0) == 1  # batch size
        
        for feature in style_features:
            assert feature.dim() == 4
            assert feature.size(0) == 1


if __name__ == "__main__":
    pytest.main([__file__])
