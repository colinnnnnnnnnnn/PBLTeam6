import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class WriterIdentificationModel(nn.Module):
    """Neural network model for writer identification from handwritten text images."""
    
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of writer classes
            backbone: Backbone architecture (resnet18, resnet34, resnet50)
            pretrained: Whether to use pretrained weights
        """
        super(WriterIdentificationModel, self).__init__()
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the classifier layers."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Pass through classifier
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x):
        """
        Extract features from the backbone (for feature analysis).
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return features

class WriterIdentificationModelV2(nn.Module):
    """Enhanced model with attention mechanism for writer identification."""
    
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True):
        """
        Initialize the enhanced model.
        
        Args:
            num_classes: Number of writer classes
            backbone: Backbone architecture
            pretrained: Whether to use pretrained weights
        """
        super(WriterIdentificationModelV2, self).__init__()
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the classifier layers."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
        
        Returns:
            Logits tensor
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classification
        logits = self.classifier(attended_features)
        
        return logits

def create_model(num_classes: int, backbone: str = "resnet18", 
                pretrained: bool = True, model_version: str = "v1") -> nn.Module:
    """
    Factory function to create a writer identification model.
    
    Args:
        num_classes: Number of writer classes
        backbone: Backbone architecture
        pretrained: Whether to use pretrained weights
        model_version: Model version ("v1" or "v2")
    
    Returns:
        Initialized model
    """
    if model_version == "v1":
        return WriterIdentificationModel(num_classes, backbone, pretrained)
    elif model_version == "v2":
        return WriterIdentificationModelV2(num_classes, backbone, pretrained)
    else:
        raise ValueError(f"Unsupported model version: {model_version}") 