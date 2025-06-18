import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import List, Tuple, Dict
import random

class WriterIdentificationDataset(Dataset):
    """Custom dataset for writer identification from handwritten text images."""
    
    def __init__(self, data_dir: Path, transform=None, is_training=True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the data directory containing writer folders
            transform: Optional transforms to apply to images
            is_training: Whether this is for training (affects augmentation)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Get all writer folders and create label mapping
        self.writer_folders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()])
        self.writer_to_label = {writer.name: idx for idx, writer in enumerate(self.writer_folders)}
        self.label_to_writer = {idx: writer.name for idx, writer in enumerate(self.writer_folders)}
        
        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for writer_folder in self.writer_folders:
            writer_name = writer_folder.name
            label = self.writer_to_label[writer_name]
            
            # Get all PNG images in the writer folder
            image_files = list(writer_folder.glob("*.png"))
            
            for image_path in image_files:
                self.image_paths.append(image_path)
                self.labels.append(label)
        
        print(f"Dataset loaded: {len(self.image_paths)} images from {len(self.writer_folders)} writers")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            # Convert PIL image to numpy array for Albumentations
            image = np.array(image)
            # Apply transforms with named arguments
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # If no transform, convert PIL to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def get_writer_info(self) -> Dict:
        """Get information about writers and their label mappings."""
        return {
            'num_writers': len(self.writer_folders),
            'writer_to_label': self.writer_to_label,
            'label_to_writer': self.label_to_writer
        }

def get_transforms(image_size: Tuple[int, int] = (224, 224), is_training: bool = True):
    """
    Get transforms for data augmentation and preprocessing.
    
    Args:
        image_size: Target image size (width, height)
        is_training: Whether transforms are for training (includes augmentation)
    
    Returns:
        Transform pipeline
    """
    if is_training:
        # Training transforms with augmentation
        transform = A.Compose([
            A.Resize(image_size[1], image_size[0]),
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = A.Compose([
            A.Resize(image_size[1], image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return transform

def create_data_loaders(data_dir: Path, batch_size: int = 32, 
                       train_split: float = 0.7, val_split: float = 0.2,
                       image_size: Tuple[int, int] = (224, 224), seed: int = 42):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        image_size: Target image size
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_info)
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create full dataset to get writer information
    full_dataset = WriterIdentificationDataset(data_dir, transform=None, is_training=False)
    dataset_info = full_dataset.get_writer_info()
    
    # Get all indices
    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)
    
    # Split indices
    train_size = int(train_split * len(all_indices))
    val_size = int(val_split * len(all_indices))
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    # Create datasets with transforms
    train_dataset = torch.utils.data.Subset(
        WriterIdentificationDataset(data_dir, get_transforms(image_size, is_training=True), is_training=True),
        train_indices
    )
    
    val_dataset = torch.utils.data.Subset(
        WriterIdentificationDataset(data_dir, get_transforms(image_size, is_training=False), is_training=False),
        val_indices
    )
    
    test_dataset = torch.utils.data.Subset(
        WriterIdentificationDataset(data_dir, get_transforms(image_size, is_training=False), is_training=False),
        test_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Data splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, dataset_info 