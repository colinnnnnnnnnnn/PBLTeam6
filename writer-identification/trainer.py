import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import time

class WriterIdentificationTrainer:
    """Trainer class for writer identification model."""
    
    def __init__(self, model: nn.Module, device: str = "cuda", 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            device: Device to train on ("cuda" or "cpu")
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, (data, targets) in enumerate(progress_bar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, save_path: Optional[Path] = None,
              early_stopping_patience: int = 10) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_path: Path to save the best model
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    self.save_model(save_path)
                    print(f"Best model saved to {save_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss
        }
        
        return history
    
    def evaluate(self, test_loader: DataLoader, label_to_writer: Dict) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            label_to_writer: Mapping from label indices to writer names
        
        Returns:
            Evaluation results dictionary
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        avg_loss = total_loss / len(test_loader)
        
        # Get unique classes present in both targets and predictions
        unique_targets = set(all_targets)
        unique_predictions = set(all_predictions)
        unique_classes = sorted(unique_targets.union(unique_predictions))
        target_names = [label_to_writer.get(cls, f"writer_{cls}") for cls in unique_classes]
        
        # Classification report
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=unique_classes)
        
        results = {
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'unique_classes': unique_classes,
            'target_names': target_names
        }
        
        return results
    
    def save_model(self, path: Path):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }, path)
    
    def load_model(self, path: Path):
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        self.learning_rates = checkpoint['learning_rates']
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Accuracy difference plot
        acc_diff = [t - v for t, v in zip(self.train_accuracies, self.val_accuracies)]
        axes[1, 1].plot(acc_diff)
        axes[1, 1].set_title('Train-Val Accuracy Difference')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, label_to_writer: Dict, 
                            save_path: Optional[Path] = None, target_names: Optional[List[str]] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(12, 10))
        
        # Use provided target names or generate from label_to_writer
        if target_names is None:
            writer_names = [label_to_writer[i] for i in range(len(label_to_writer))]
        else:
            writer_names = target_names
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=writer_names, yticklabels=writer_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 