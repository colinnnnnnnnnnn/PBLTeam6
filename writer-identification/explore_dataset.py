#!/usr/bin/env python3
"""
Dataset exploration utility for writer identification project.
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from PIL import Image
import argparse

def explore_dataset(data_dir: Path):
    """
    Explore and analyze the dataset structure.
    
    Args:
        data_dir: Path to the data directory
    """
    print("="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist!")
        return
    
    # Get all writer folders
    writer_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    
    if not writer_folders:
        print(f"Error: No writer folders found in {data_dir}")
        return
    
    print(f"Found {len(writer_folders)} writer folders")
    print(f"Data directory: {data_dir.absolute()}")
    print()
    
    # Analyze each writer folder
    writer_stats = {}
    total_images = 0
    
    for writer_folder in writer_folders:
        writer_name = writer_folder.name
        
        # Count images
        image_files = list(writer_folder.glob("*.png"))
        num_images = len(image_files)
        total_images += num_images
        
        writer_stats[writer_name] = {
            'num_images': num_images,
            'image_files': [f.name for f in image_files]
        }
        
        print(f"Writer {writer_name}: {num_images} images")
    
    print(f"\nTotal images: {total_images}")
    print(f"Average images per writer: {total_images / len(writer_folders):.1f}")
    
    # Analyze image distribution
    image_counts = [stats['num_images'] for stats in writer_stats.values()]
    print(f"Min images per writer: {min(image_counts)}")
    print(f"Max images per writer: {max(image_counts)}")
    print(f"Standard deviation: {np.std(image_counts):.1f}")
    
    # Check for class imbalance
    print("\nClass Balance Analysis:")
    balanced_threshold = 0.8  # 80% of average
    avg_images = np.mean(image_counts)
    min_threshold = avg_images * balanced_threshold
    
    imbalanced_writers = []
    for writer_name, stats in writer_stats.items():
        if stats['num_images'] < min_threshold:
            imbalanced_writers.append(writer_name)
    
    if imbalanced_writers:
        print(f"⚠️  {len(imbalanced_writers)} writers have fewer than {min_threshold:.1f} images:")
        for writer in imbalanced_writers:
            print(f"   - {writer}: {writer_stats[writer]['num_images']} images")
    else:
        print("✅ Dataset appears to be relatively balanced")
    
    # Analyze image sizes
    print("\nAnalyzing image sizes...")
    image_sizes = []
    sample_writer = writer_folders[0]
    sample_images = list(sample_writer.glob("*.png"))[:5]  # Sample first 5 images
    
    for img_path in sample_images:
        try:
            with Image.open(img_path) as img:
                image_sizes.append(img.size)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    if image_sizes:
        print(f"Sample image sizes: {image_sizes}")
        avg_width = np.mean([size[0] for size in image_sizes])
        avg_height = np.mean([size[1] for size in image_sizes])
        print(f"Average image size: {avg_width:.0f} x {avg_height:.0f}")
    
    # Create visualizations
    create_dataset_visualizations(writer_stats, data_dir)
    
    # Save dataset info
    dataset_info = {
        'num_writers': len(writer_folders),
        'total_images': total_images,
        'writer_stats': writer_stats,
        'image_counts': image_counts,
        'avg_images_per_writer': total_images / len(writer_folders),
        'min_images': min(image_counts),
        'max_images': max(image_counts),
        'std_images': float(np.std(image_counts))
    }
    
    info_path = data_dir.parent / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset info saved to: {info_path}")
    
    return dataset_info

def create_dataset_visualizations(writer_stats: dict, data_dir: Path):
    """Create visualizations of the dataset."""
    
    # Create output directory for plots
    plots_dir = data_dir.parent / "dataset_analysis"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Images per writer bar plot
    plt.figure(figsize=(15, 8))
    writers = list(writer_stats.keys())
    image_counts = [stats['num_images'] for stats in writer_stats.values()]
    
    plt.bar(range(len(writers)), image_counts)
    plt.title('Number of Images per Writer', fontsize=16)
    plt.xlabel('Writer ID', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(range(len(writers)), writers, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "images_per_writer.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(image_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Images per Writer', fontsize=16)
    plt.xlabel('Number of Images', fontsize=12)
    plt.ylabel('Number of Writers', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "images_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sample images grid
    plt.figure(figsize=(15, 10))
    sample_writers = list(writer_stats.keys())[:12]  # Show first 12 writers
    
    for i, writer_name in enumerate(sample_writers):
        writer_folder = data_dir / writer_name
        image_files = list(writer_folder.glob("*.png"))
        
        if image_files:
            # Load first image
            img_path = image_files[0]
            try:
                with Image.open(img_path) as img:
                    plt.subplot(3, 4, i + 1)
                    plt.imshow(img, cmap='gray')
                    plt.title(f'{writer_name}\n({writer_stats[writer_name]["num_images"]} images)', fontsize=10)
                    plt.axis('off')
            except Exception as e:
                plt.subplot(3, 4, i + 1)
                plt.text(0.5, 0.5, f'Error loading\n{writer_name}', ha='center', va='center')
                plt.axis('off')
    
    plt.suptitle('Sample Images from Each Writer', fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / "sample_images.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {plots_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Explore Writer Identification Dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    explore_dataset(data_dir)

if __name__ == "__main__":
    main() 