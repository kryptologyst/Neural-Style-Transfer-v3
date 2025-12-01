#!/usr/bin/env python3
"""Data generation script for Neural Style Transfer."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nst import SampleDatasetGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate sample datasets for Neural Style Transfer")
    parser.add_argument("--content-dir", type=str, default="data/content", help="Content images directory")
    parser.add_argument("--style-dir", type=str, default="data/style", help="Style images directory")
    parser.add_argument("--num-content", type=int, default=10, help="Number of content images")
    parser.add_argument("--num-style", type=int, default=10, help="Number of style images")
    
    return parser.parse_args()


def main():
    """Main data generation function."""
    args = parse_args()
    
    print("Generating sample datasets for Neural Style Transfer...")
    
    # Create directories
    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    
    content_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Content directory: {content_dir}")
    print(f"Style directory: {style_dir}")
    
    # Generate content images
    print(f"\nGenerating {args.num_content} content images...")
    SampleDatasetGenerator.create_sample_content_images(str(content_dir), args.num_content)
    
    # Generate style images
    print(f"\nGenerating {args.num_style} style images...")
    SampleDatasetGenerator.create_sample_style_images(str(style_dir), args.num_style)
    
    print("\nDataset generation completed!")
    print(f"Content images: {len(list(content_dir.glob('*.jpg')))}")
    print(f"Style images: {len(list(style_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
