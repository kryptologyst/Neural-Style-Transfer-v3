#!/usr/bin/env python3
"""Training script for Neural Style Transfer."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from nst import (
    NeuralStyleTransfer, 
    StyleTransferDataset, 
    Config, 
    StyleTransferEvaluator,
    set_seed,
    get_device
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Neural Style Transfer model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--content-dir", type=str, help="Content images directory")
    parser.add_argument("--style-dir", type=str, help="Style images directory")
    parser.add_argument("--output-dir", type=str, default="assets", help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--style-weight", type=float, help="Style loss weight")
    parser.add_argument("--content-weight", type=float, help="Content loss weight")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.content_dir:
        config.set("data.content_dir", args.content_dir)
    if args.style_dir:
        config.set("data.style_dir", args.style_dir)
    if args.output_dir:
        config.set("data.output_dir", args.output_dir)
    if args.epochs:
        config.set("training.num_epochs", args.epochs)
    if args.lr:
        config.set("training.learning_rate", args.lr)
    if args.style_weight:
        config.set("training.style_weight", args.style_weight)
    if args.content_weight:
        config.set("training.content_weight", args.content_weight)
    if args.device:
        config.set("system.device", args.device)
    if args.seed:
        config.set("system.seed", args.seed)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # Set up device and seed
    device = get_device() if config.get("system.device") == "auto" else torch.device(config.get("system.device"))
    set_seed(config.get("system.seed"))
    
    print(f"Using device: {device}")
    print(f"Configuration: {config.get('model.type')} model")
    
    # Create output directory
    output_dir = Path(config.get("data.output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = NeuralStyleTransfer(
        device=device,
        seed=config.get("system.seed"),
        style_weight=config.get("training.style_weight"),
        content_weight=config.get("training.content_weight")
    )
    
    # Load sample images for training
    content_dir = Path(config.get("data.content_dir"))
    style_dir = Path(config.get("data.style_dir"))
    
    content_files = list(content_dir.glob("*.jpg")) + list(content_dir.glob("*.png"))
    style_files = list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png"))
    
    if not content_files or not style_files:
        print("No images found in content or style directories")
        print("Please add images to the directories or run the data generation script")
        return
    
    # Use first content and style images for training
    content_path = str(content_files[0])
    style_path = str(style_files[0])
    
    print(f"Training with content: {content_path}")
    print(f"Training with style: {style_path}")
    
    # Load images
    model.load_images(content_path, style_path, config.get("model.max_size"))
    
    # Training loop
    num_epochs = config.get("training.num_epochs")
    learning_rate = config.get("training.learning_rate")
    
    print(f"Starting training for {num_epochs} epochs...")
    
    try:
        stylized = model.transfer_style(num_epochs, learning_rate)
        
        # Save result
        output_path = output_dir / "trained_stylized.jpg"
        model.save_result(str(output_path))
        print(f"Training completed! Saved result to {output_path}")
        
        # Run evaluation if requested
        if args.eval:
            print("Running evaluation...")
            evaluator = StyleTransferEvaluator(device)
            
            # Load images for evaluation
            content_tensor = model.content_image
            style_tensor = model.style_image
            stylized_tensor = model.target_image.detach()
            
            # Evaluate
            metrics = evaluator.evaluate_batch(content_tensor, style_tensor, stylized_tensor)
            
            print("Evaluation Results:")
            print(f"  Content Preservation: {metrics['content_preservation']:.4f}")
            print(f"  Style Similarity: {metrics['style_similarity']:.4f}")
            print(f"  Perceptual Distance: {metrics['perceptual_distance']:.4f}")
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save current result
        output_path = output_dir / "interrupted_stylized.jpg"
        model.save_result(str(output_path))
        print(f"Saved current result to {output_path}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
