"""
Main training script for writer identification model.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
from pathlib import Path
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import create_data_loaders
from model import create_model
from trainer import WriterIdentificationTrainer

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Writer Identification Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Path to output directory')
    parser.add_argument('--model_name', type=str, default='writer_identification_model', help='Model name')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'], help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Image size (width height)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_version', type=str, default='v1', choices=['v1', 'v2'], help='Model version')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='Do not use pretrained backbone')
    parser.set_defaults(pretrained=True)
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        train_split=0.7,
        val_split=0.2,
        image_size=tuple(args.image_size),
        seed=args.seed
    )
    
    num_classes = dataset_info['num_writers']
    print(f"Number of writers: {num_classes}")
    print(f"Writer labels: {list(dataset_info['writer_to_label'].keys())}")
    
    # Create model
    print(f"Creating model with backbone: {args.backbone}")
    model = create_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        model_version=args.model_version
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = WriterIdentificationTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=1e-4
    )
    
    # Training configuration
    model_save_path = output_dir / f"{args.model_name}.pth"
    history_save_path = output_dir / f"{args.model_name}_history.json"
    plots_save_path = output_dir / f"{args.model_name}_plots.png"
    
    # Train the model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_path=model_save_path,
        early_stopping_patience=10
    )
    
    # Save training history
    with open(history_save_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model metadata
    metadata_save_path = output_dir / f"{args.model_name}_metadata.json"
    metadata = {
        'num_classes': num_classes,
        'backbone': args.backbone,
        'model_version': args.model_version,
        'image_size': args.image_size,
        'writer_to_label': dataset_info['writer_to_label'],
        'label_to_writer': dataset_info['label_to_writer']
    }
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Plot training history
    print("Plotting training history...")
    trainer.plot_training_history(save_path=plots_save_path)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_loader, dataset_info['label_to_writer'])
    
    # Save test results
    test_results_save_path = output_dir / f"{args.model_name}_test_results.json"
    with open(test_results_save_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'accuracy': float(test_results['accuracy']),
            'avg_loss': float(test_results['avg_loss']),
            'classification_report': test_results['classification_report'],
            'confusion_matrix': test_results['confusion_matrix'].tolist(),
            'predictions': [int(p) for p in test_results['predictions']],
            'targets': [int(t) for t in test_results['targets']],
            'unique_classes': [int(cls) for cls in test_results.get('unique_classes', [])],
            'target_names': test_results.get('target_names', [])
        }
        json.dump(serializable_results, f, indent=2)
    
    # Plot confusion matrix
    confusion_matrix_save_path = output_dir / f"{args.model_name}_confusion_matrix.png"
    trainer.plot_confusion_matrix(
        test_results['confusion_matrix'],
        dataset_info['label_to_writer'],
        save_path=confusion_matrix_save_path,
        target_names=test_results.get('target_names')
    )
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    print(f"Test accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"Test loss: {test_results['avg_loss']:.4f}")
    print(f"\nModel saved to: {model_save_path}")
    print(f"Training history saved to: {history_save_path}")
    print(f"Test results saved to: {test_results_save_path}")
    print(f"Plots saved to: {plots_save_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_save_path}")
    
    # Print classification report
    print("\nClassification Report:")
    print(test_results['classification_report'])

if __name__ == "__main__":
    main() 