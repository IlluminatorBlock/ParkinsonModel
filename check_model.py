import os
import sys
import torch
import json
import numpy as np
from pathlib import Path

# Define paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = ROOT_DIR / 'models' / 'pretrained'
best_model_path = os.path.join(MODELS_DIR, 'best_model.pt')

print(f"Checking model at {best_model_path}")

try:
    checkpoint = torch.load(best_model_path, map_location='cpu')
    print(f"Model loaded successfully")
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        print("\nModel state dict exists")
        model_size = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        print(f"Model has {model_size:,} parameters")
    
    if 'val_metrics' in checkpoint:
        metrics = checkpoint['val_metrics']
        print("\nValidation Metrics:")
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                print(f"{k}: {v:.4f}")
        
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            print("\nConfusion Matrix:")
            print(cm)
    else:
        print("\nNo validation metrics found in the checkpoint")
    
    print("\nEpoch info:")
    print(f"Epoch: {checkpoint.get('epoch', 'Not found')}")
    print(f"Train loss: {checkpoint.get('train_loss', 'Not found')}")
    print(f"Val loss: {checkpoint.get('val_loss', 'Not found')}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    
    # Check if the file exists
    if os.path.exists(best_model_path):
        print(f"File exists, size: {os.path.getsize(best_model_path) / 1024 / 1024:.2f} MB")
    else:
        print("File does not exist")
        
    # List all model files
    print("\nAvailable model files:")
    for file in os.listdir(MODELS_DIR):
        print(f" - {file} ({os.path.getsize(os.path.join(MODELS_DIR, file)) / 1024 / 1024:.2f} MB)") 