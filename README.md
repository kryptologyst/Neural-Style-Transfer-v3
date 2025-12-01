# Neural Style Transfer

A production-ready implementation of Neural Style Transfer using PyTorch. This project provides multiple style transfer techniques including the original VGG-based approach, fast feedforward models, and multi-scale processing.

## Features

- **Multiple Model Architectures**: VGG19-based iterative optimization, fast feedforward models, and multi-scale processing
- **Modern PyTorch Stack**: Built with PyTorch 2.0+, supporting CUDA, MPS (Apple Silicon), and CPU
- **Comprehensive Evaluation**: Content preservation, style similarity, and perceptual quality metrics
- **Interactive Demo**: Streamlit web interface for easy experimentation
- **Production Ready**: Proper configuration management, logging, and testing
- **Reproducible**: Deterministic seeding and comprehensive documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Neural-Style-Transfer-v3
cd Neural-Style-Transfer-v3

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```python
from nst import NeuralStyleTransfer, Config

# Load configuration
config = Config("configs/default.yaml")

# Initialize model
nst = NeuralStyleTransfer()

# Load images
nst.load_images("path/to/content.jpg", "path/to/style.jpg")

# Perform style transfer
result = nst.transfer_style(num_epochs=500)

# Save result
nst.save_result("output.jpg")
```

### Command Line Interface

```bash
# Generate sample data
python scripts/generate_data.py --num-content 10 --num-style 10

# Train a model
python scripts/train.py --config configs/default.yaml --epochs 500

# Generate samples
python scripts/sample.py --content content.jpg --style style.jpg --output result.jpg

# Run interactive demo
streamlit run demo/streamlit_app.py
```

## Model Architectures

### 1. VGG19-based Iterative Optimization
The original Neural Style Transfer approach using VGG19 feature extraction and iterative optimization.

**Pros:**
- High quality results
- Well-established technique
- Good balance of content and style

**Cons:**
- Slow inference (requires optimization)
- Memory intensive

### 2. Fast Neural Style Transfer
Feedforward encoder-decoder architecture for real-time style transfer.

**Pros:**
- Fast inference
- Single forward pass
- Good for real-time applications

**Cons:**
- Requires training on style-specific datasets
- May have lower quality than iterative methods

### 3. Multi-Scale Style Transfer
Processes images at multiple scales for improved quality and detail preservation.

**Pros:**
- Better detail preservation
- Improved quality at high resolutions
- Robust to different image sizes

**Cons:**
- More computationally expensive
- Complex training procedure

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
# Model settings
model:
  type: vgg19  # 'vgg19', 'fast_nst', 'multi_scale'
  pretrained: true
  max_size: 512

# Training settings
training:
  num_epochs: 500
  learning_rate: 1.0
  style_weight: 1000000.0
  content_weight: 1.0

# System settings
system:
  device: auto  # 'auto', 'cuda', 'mps', 'cpu'
  seed: 42
```

## Evaluation Metrics

The project includes comprehensive evaluation metrics:

- **Content Preservation**: Measures how well the content structure is preserved
- **Style Similarity**: Quantifies how closely the style matches the reference
- **Perceptual Distance**: Uses LPIPS to measure perceptual similarity
- **FID Score**: Fréchet Inception Distance for overall quality assessment

## Interactive Demo

Launch the Streamlit demo for an interactive experience:

```bash
streamlit run demo/streamlit_app.py
```

The demo provides:
- Real-time style transfer
- Adjustable parameters
- Image upload and download
- Performance monitoring

## API Reference

### Core Classes

#### `NeuralStyleTransfer`
Main class for iterative style transfer using VGG19.

```python
nst = NeuralStyleTransfer(
    device=None,
    seed=42,
    style_weight=1e6,
    content_weight=1.0
)
```

#### `StyleTransferSampler`
High-level interface for style transfer inference.

```python
sampler = StyleTransferSampler(config)
result = sampler.transfer_style(content_path, style_path)
```

#### `StyleTransferEvaluator`
Comprehensive evaluation of style transfer results.

```python
evaluator = StyleTransferEvaluator(device)
metrics = evaluator.evaluate_batch(content, style, stylized)
```

### Configuration

#### `Config`
Configuration management with YAML support.

```python
config = Config("configs/default.yaml")
config.set("training.num_epochs", 1000)
config.save_config("custom_config.yaml")
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/ scripts/

# Lint code
ruff check src/ tests/ scripts/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA or MPS for faster processing
2. **Batch Processing**: Process multiple images simultaneously
3. **Image Resolution**: Lower resolutions process faster but may reduce quality
4. **Model Selection**: Choose the right model for your use case
5. **Parameter Tuning**: Adjust style/content weights for optimal results

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image resolution
2. **Slow Processing**: Ensure GPU is being used, consider using fast models
3. **Poor Quality**: Adjust style/content weights, increase epochs
4. **Import Errors**: Ensure all dependencies are installed correctly

### Performance Optimization

- Use mixed precision training for faster processing
- Enable model compilation with `torch.compile()`
- Optimize data loading with multiple workers
- Use appropriate image sizes for your hardware

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neural_style_transfer,
  title={Neural Style Transfer: A Modern PyTorch Implementation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Neural-Style-Transfer-v3}
}
```

## Acknowledgments

- Original Neural Style Transfer paper by Gatys et al.
- PyTorch team for the excellent framework
- VGG19 pre-trained models from torchvision
- The open-source community for various components and inspiration
# Neural-Style-Transfer-v3
