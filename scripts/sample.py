#!/usr/bin/env python3
"""Sampling script for Neural Style Transfer."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import random
from PIL import Image

from nst import (
    StyleTransferSampler,
    Config,
    create_sample_grid,
    random_style_transfer,
    set_seed
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate samples with Neural Style Transfer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--content", type=str, help="Content image path")
    parser.add_argument("--style", type=str, help="Style image path")
    parser.add_argument("--output", type=str, help="Output image path")
    parser.add_argument("--content-dir", type=str, help="Content images directory")
    parser.add_argument("--style-dir", type=str, help="Style images directory")
    parser.add_argument("--output-dir", type=str, default="assets", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of random samples")
    parser.add_argument("--epochs", type=int, help="Number of optimization epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--interpolate", action="store_true", help="Interpolate between styles")
    parser.add_argument("--batch", action="store_true", help="Process batch of images")
    
    return parser.parse_args()


def main():
    """Main sampling function."""
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
    if args.device:
        config.set("system.device", args.device)
    if args.seed:
        config.set("system.seed", args.seed)
    
    # Set seed for reproducibility
    set_seed(config.get("system.seed"))
    
    # Initialize sampler
    sampler = StyleTransferSampler(config)
    
    print(f"Using device: {sampler.device}")
    print(f"Model type: {config.get('model.type')}")
    
    # Create output directory
    output_dir = Path(config.get("data.output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.batch:
        # Batch processing
        content_dir = Path(config.get("data.content_dir"))
        style_dir = Path(config.get("data.style_dir"))
        
        content_files = list(content_dir.glob("*.jpg")) + list(content_dir.glob("*.png"))
        style_files = list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png"))
        
        if not content_files or not style_files:
            print("No images found in content or style directories")
            return
        
        print(f"Processing {len(content_files)} content images with {len(style_files)} style images")
        
        # Process all combinations
        results = []
        for i, content_path in enumerate(content_files):
            for j, style_path in enumerate(style_files):
                output_path = output_dir / f"batch_{i:03d}_{j:03d}.jpg"
                
                try:
                    stylized = sampler.transfer_style(
                        str(content_path), 
                        str(style_path), 
                        str(output_path)
                    )
                    results.append(stylized)
                    print(f"Generated: {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {content_path} + {style_path}: {e}")
                    continue
        
        print(f"Batch processing completed! Generated {len(results)} images")
        
    elif args.interpolate:
        # Style interpolation
        content_dir = Path(config.get("data.content_dir"))
        style_dir = Path(config.get("data.style_dir"))
        
        content_files = list(content_dir.glob("*.jpg")) + list(content_dir.glob("*.png"))
        style_files = list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png"))
        
        if not content_files or not style_files:
            print("No images found for interpolation")
            return
        
        # Use first content image and first two style images
        content_path = str(content_files[0])
        style_paths = [str(style_files[0]), str(style_files[1]) if len(style_files) > 1 else str(style_files[0])]
        
        print(f"Interpolating styles for content: {content_path}")
        print(f"Style images: {style_paths}")
        
        results = sampler.interpolate_styles(
            content_path, 
            style_paths, 
            str(output_dir),
            num_interpolations=args.num_samples
        )
        
        print(f"Style interpolation completed! Generated {len(results)} images")
        
    elif args.content and args.style:
        # Single image processing
        output_path = args.output or str(output_dir / "stylized_output.jpg")
        
        print(f"Processing content: {args.content}")
        print(f"Processing style: {args.style}")
        print(f"Output: {output_path}")
        
        try:
            stylized = sampler.transfer_style(
                args.content, 
                args.style, 
                output_path
            )
            print(f"Style transfer completed! Saved to {output_path}")
            
        except Exception as e:
            print(f"Error during style transfer: {e}")
            raise
            
    else:
        # Random sampling
        content_dir = config.get("data.content_dir")
        style_dir = config.get("data.style_dir")
        
        print(f"Generating {args.num_samples} random style transfer samples")
        
        random_style_transfer(
            content_dir, 
            style_dir, 
            str(output_dir),
            config,
            args.num_samples
        )
        
        print(f"Random sampling completed! Check {output_dir}")


if __name__ == "__main__":
    main()
