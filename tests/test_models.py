"""Unit tests for Neural Style Transfer models."""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nst.models import (
    AdaIN,
    FastNSTModel,
    ResidualBlock,
    StyleEncoder,
    ContentEncoder,
    MultiScaleStyleTransfer,
    PerceptualLoss,
    TotalVariationLoss
)


class TestAdaIN:
    """Test AdaIN layer."""
    
    def test_adain_initialization(self):
        """Test AdaIN initialization."""
        adain = AdaIN()
        assert isinstance(adain, AdaIN)
        assert adain.eps == 1e-5
    
    def test_adain_forward(self):
        """Test AdaIN forward pass."""
        adain = AdaIN()
        
        content = torch.randn(1, 64, 32, 32)
        style = torch.randn(1, 64, 32, 32)
        
        output = adain(content, style)
        
        assert output.shape == content.shape
        assert isinstance(output, torch.Tensor)
    
    def test_adain_custom_eps(self):
        """Test AdaIN with custom epsilon."""
        adain = AdaIN(eps=1e-3)
        assert adain.eps == 1e-3


class TestResidualBlock:
    """Test ResidualBlock."""
    
    def test_residual_block_initialization(self):
        """Test ResidualBlock initialization."""
        block = ResidualBlock(64)
        
        assert isinstance(block, ResidualBlock)
        assert isinstance(block.conv1, nn.Conv2d)
        assert isinstance(block.conv2, nn.Conv2d)
        assert isinstance(block.bn1, nn.InstanceNorm2d)
        assert isinstance(block.bn2, nn.InstanceNorm2d)
    
    def test_residual_block_forward(self):
        """Test ResidualBlock forward pass."""
        block = ResidualBlock(64)
        
        x = torch.randn(1, 64, 32, 32)
        output = block(x)
        
        assert output.shape == x.shape
        assert isinstance(output, torch.Tensor)
    
    def test_residual_connection(self):
        """Test that residual connection works."""
        block = ResidualBlock(64)
        
        x = torch.randn(1, 64, 32, 32)
        output = block(x)
        
        # Output should be different from input due to non-linearities
        assert not torch.allclose(output, x)


class TestFastNSTModel:
    """Test FastNSTModel."""
    
    def test_fast_nst_initialization(self):
        """Test FastNSTModel initialization."""
        model = FastNSTModel()
        
        assert isinstance(model, FastNSTModel)
        assert isinstance(model.encoder, nn.Sequential)
        assert isinstance(model.decoder, nn.Sequential)
        assert len(model.residual_blocks) == 5
    
    def test_fast_nst_forward(self):
        """Test FastNSTModel forward pass."""
        model = FastNSTModel()
        
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        
        assert output.shape == x.shape
        assert isinstance(output, torch.Tensor)
    
    def test_fast_nst_custom_channels(self):
        """Test FastNSTModel with custom channels."""
        model = FastNSTModel(input_channels=1, output_channels=1)
        
        x = torch.randn(1, 1, 256, 256)
        output = model(x)
        
        assert output.shape == x.shape


class TestStyleEncoder:
    """Test StyleEncoder."""
    
    def test_style_encoder_initialization(self):
        """Test StyleEncoder initialization."""
        encoder = StyleEncoder()
        
        assert isinstance(encoder, StyleEncoder)
        assert hasattr(encoder, 'features')
    
    def test_style_encoder_forward(self):
        """Test StyleEncoder forward pass."""
        encoder = StyleEncoder()
        
        x = torch.randn(1, 3, 224, 224)
        features = encoder(x)
        
        assert isinstance(features, list)
        assert len(features) == 5  # Should extract 5 style features
    
    def test_style_encoder_frozen_parameters(self):
        """Test that StyleEncoder parameters are frozen."""
        encoder = StyleEncoder()
        
        for param in encoder.features.parameters():
            assert not param.requires_grad


class TestContentEncoder:
    """Test ContentEncoder."""
    
    def test_content_encoder_initialization(self):
        """Test ContentEncoder initialization."""
        encoder = ContentEncoder()
        
        assert isinstance(encoder, ContentEncoder)
        assert hasattr(encoder, 'features')
    
    def test_content_encoder_forward(self):
        """Test ContentEncoder forward pass."""
        encoder = ContentEncoder()
        
        x = torch.randn(1, 3, 224, 224)
        features = encoder(x)
        
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 4
    
    def test_content_encoder_frozen_parameters(self):
        """Test that ContentEncoder parameters are frozen."""
        encoder = ContentEncoder()
        
        for param in encoder.features.parameters():
            assert not param.requires_grad


class TestMultiScaleStyleTransfer:
    """Test MultiScaleStyleTransfer."""
    
    def test_multi_scale_initialization(self):
        """Test MultiScaleStyleTransfer initialization."""
        model = MultiScaleStyleTransfer()
        
        assert isinstance(model, MultiScaleStyleTransfer)
        assert hasattr(model, 'scales')
        assert hasattr(model, 'models')
        assert len(model.models) == len(model.scales)
    
    def test_multi_scale_forward(self):
        """Test MultiScaleStyleTransfer forward pass."""
        model = MultiScaleStyleTransfer()
        
        x = torch.randn(1, 3, 256, 256)
        outputs = model(x)
        
        assert isinstance(outputs, list)
        assert len(outputs) == len(model.scales)
        
        for output in outputs:
            assert output.shape == x.shape
    
    def test_multi_scale_custom_scales(self):
        """Test MultiScaleStyleTransfer with custom scales."""
        scales = [128, 256]
        model = MultiScaleStyleTransfer(scales)
        
        assert model.scales == scales
        assert len(model.models) == len(scales)


class TestPerceptualLoss:
    """Test PerceptualLoss."""
    
    def test_perceptual_loss_initialization(self):
        """Test PerceptualLoss initialization."""
        loss = PerceptualLoss()
        
        assert isinstance(loss, PerceptualLoss)
        assert hasattr(loss, 'features')
        assert hasattr(loss, 'feature_layers')
    
    def test_perceptual_loss_forward(self):
        """Test PerceptualLoss forward pass."""
        loss = PerceptualLoss()
        
        x = torch.randn(1, 3, 224, 224)
        target = torch.randn(1, 3, 224, 224)
        
        loss_value = loss(x, target)
        
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.item() >= 0
    
    def test_perceptual_loss_identical_inputs(self):
        """Test PerceptualLoss with identical inputs."""
        loss = PerceptualLoss()
        
        x = torch.randn(1, 3, 224, 224)
        loss_value = loss(x, x)
        
        assert torch.allclose(loss_value, torch.tensor(0.0))
    
    def test_perceptual_loss_frozen_parameters(self):
        """Test that PerceptualLoss parameters are frozen."""
        loss = PerceptualLoss()
        
        for param in loss.features.parameters():
            assert not param.requires_grad


class TestTotalVariationLoss:
    """Test TotalVariationLoss."""
    
    def test_tv_loss_initialization(self):
        """Test TotalVariationLoss initialization."""
        loss = TotalVariationLoss()
        
        assert isinstance(loss, TotalVariationLoss)
    
    def test_tv_loss_forward(self):
        """Test TotalVariationLoss forward pass."""
        loss = TotalVariationLoss()
        
        x = torch.randn(1, 3, 64, 64)
        loss_value = loss(x)
        
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.item() >= 0
    
    def test_tv_loss_constant_input(self):
        """Test TotalVariationLoss with constant input."""
        loss = TotalVariationLoss()
        
        # Constant tensor should have zero TV loss
        x = torch.ones(1, 3, 64, 64)
        loss_value = loss(x)
        
        assert torch.allclose(loss_value, torch.tensor(0.0))
    
    def test_tv_loss_noisy_input(self):
        """Test TotalVariationLoss with noisy input."""
        loss = TotalVariationLoss()
        
        # Noisy tensor should have high TV loss
        x = torch.randn(1, 3, 64, 64)
        loss_value = loss(x)
        
        assert loss_value.item() > 0


if __name__ == "__main__":
    pytest.main([__file__])
