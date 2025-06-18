"""
Prediction script for writer identification model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
from pathlib import Path
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from model import create_model
from typing import List, Dict, Tuple
import importlib.util

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from model import create_model
from dataset import get_transforms

class WriterIdentifier:
    """Class for making predictions with a trained writer identification model."""
    
    def __init__(self, model_path: Path, device: str = "auto"):
        """
        Initialize the writer identifier.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to use for inference
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Load model and metadata
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file {self.model_path} not found")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Try to load metadata from the same directory
        metadata_path = self.model_path.parent / f"{self.model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Try alternative metadata path (for quick start models)
            alt_metadata_path = self.model_path.parent / "quick_start_model_metadata.json"
            if alt_metadata_path.exists():
                with open(alt_metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                # Default metadata (you might need to adjust this based on your training)
                self.metadata = {
                    'num_classes': 50,  # Adjust based on your dataset
                    'backbone': 'resnet18',
                    'model_version': 'v1',
                    'image_size': [224, 224]
                }
        
        # Create model with correct number of classes
        self.model = create_model(
            num_classes=self.metadata['num_classes'],
            backbone=self.metadata['backbone'],
            pretrained=False,
            model_version=self.metadata['model_version']
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Model metadata: {self.metadata}")
    
    def predict_image(self, image_path: Path, top_k: int = 5) -> List[Dict]:
        """
        Predict writer for a single image.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
        
        Returns:
            List of dictionaries with prediction information
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        transform = get_transforms(
            image_size=tuple(self.metadata['image_size']), 
            is_training=False
        )
        
        # Convert PIL image to numpy array for Albumentations
        image = np.array(image)
        # Apply transforms with named arguments
        transformed = transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.metadata['num_classes']))
        
        # Format results
        predictions = []
        for i in range(len(top_indices[0])):
            pred = {
                'writer_id': f"writer_{top_indices[0][i].item():03d}",
                'confidence': top_probs[0][i].item(),
                'rank': i + 1
            }
            predictions.append(pred)
        
        return predictions
    
    def predict_batch(self, image_paths: List[Path], top_k: int = 5) -> List[List[Dict]]:
        """
        Predict writer for multiple images.
        
        Args:
            image_paths: List of paths to image files
            top_k: Number of top predictions to return per image
        
        Returns:
            List of prediction lists for each image
        """
        results = []
        
        for image_path in image_paths:
            try:
                predictions = self.predict_image(image_path, top_k)
                results.append({
                    'image_path': str(image_path),
                    'predictions': predictions,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'image_path': str(image_path),
                    'predictions': [],
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, directory_path: Path, top_k: int = 5, 
                         extensions: List[str] = None) -> List[Dict]:
        """
        Predict writer for all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            top_k: Number of top predictions to return per image
            extensions: List of file extensions to process (default: ['.png', '.jpg', '.jpeg'])
        
        Returns:
            List of prediction results for each image
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg']
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory_path.glob(f"*{ext}"))
            image_paths.extend(directory_path.glob(f"*{ext.upper()}"))
        
        if not image_paths:
            print(f"No image files found in {directory_path}")
            return []
        
        print(f"Found {len(image_paths)} images to process")
        return self.predict_batch(image_paths, top_k)

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict Writer from Handwritten Text Images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to image file or directory')
    parser.add_argument('--output', type=str, default='predictions.json', help='Output file for predictions')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Initialize writer identifier
    try:
        identifier = WriterIdentifier(args.model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Determine input type
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image prediction
        print(f"Predicting writer for single image: {input_path}")
        predictions = identifier.predict_image(input_path, args.top_k)
        
        results = {
            'image_path': str(input_path),
            'predictions': predictions
        }
        
    elif input_path.is_dir():
        # Directory prediction
        print(f"Predicting writers for images in directory: {input_path}")
        results = identifier.predict_directory(input_path, args.top_k)
        
    else:
        print(f"Input path {input_path} does not exist")
        return
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print(f"\nPredictions saved to: {output_path}")
    
    if isinstance(results, dict):
        # Single image result
        print(f"\nTop {args.top_k} predictions for {results['image_path']}:")
        for pred in results['predictions']:
            print(f"  {pred['rank']}. Writer {pred['writer_id']}: {pred['confidence']:.3f}")
    else:
        # Multiple image results
        print(f"\nProcessed {len(results)} images")
        for result in results:
            if result['status'] == 'success':
                print(f"\n{result['image_path']}:")
                for pred in result['predictions'][:3]:  # Show top 3
                    print(f"  {pred['rank']}. Writer {pred['writer_id']}: {pred['confidence']:.3f}")
            else:
                print(f"\n{result['image_path']}: ERROR - {result['error']}")

if __name__ == "__main__":
    main()

# Add dynamic import for WriterIdentifier
writer_id_path = os.path.join(os.path.dirname(__file__), 'writer-identification', 'predict.py')
WriterIdentifier = None
if os.path.exists(writer_id_path):
    spec = importlib.util.spec_from_file_location("writer_identification_predict", writer_id_path)
    writer_id_module = importlib.util.module_from_spec(spec)
    sys.modules["writer_identification_predict"] = writer_id_module
    try:
        spec.loader.exec_module(writer_id_module)
        WriterIdentifier = getattr(writer_id_module, "WriterIdentifier", None)
    except Exception as e:
        print("Error importing WriterIdentifier:", e)
        import traceback; traceback.print_exc()
        WriterIdentifier = None 