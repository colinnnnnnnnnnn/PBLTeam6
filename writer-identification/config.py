import os
from pathlib import Path

class Config:
    # Data paths
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("outputs")
    MODELS_DIR = Path("models")
    
    # Model parameters
    NUM_CLASSES = None  # Will be set dynamically based on dataset
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Training parameters
    DEVICE = "cuda"  # or "cpu"
    SEED = 42
    
    # Data augmentation
    AUGMENTATION_PROB = 0.5
    
    # Model architecture
    BACKBONE = "resnet18"  # Options: resnet18, resnet34, resnet50
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        
    @classmethod
    def get_model_path(cls, model_name):
        """Get the path for saving/loading a model."""
        return cls.MODELS_DIR / f"{model_name}.pth" 