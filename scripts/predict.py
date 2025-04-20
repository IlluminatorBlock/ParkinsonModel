#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction script for Parkinson's disease detection from MRI scans.
"""

import os
import sys
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_scan(file_path, target_shape=(128, 128, 128)):
    """Preprocess a single MRI scan for prediction."""
    # Load the scan
    print(f"Loading scan: {file_path}")
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Normalize
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    # Resize if needed
    if data.shape != target_shape:
        from scipy.ndimage import zoom
        zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
        data = zoom(data, zoom_factors, order=1)
    
    # Convert to torch tensor and add batch & channel dimensions
    data_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
    
    return data_tensor

def load_model(model_path, device):
    """Load the trained model."""
    print(f"Loading model from: {model_path}")
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict(model, input_tensor, device):
    """Run prediction on the input tensor."""
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item(), probs[0, 1].item()  # Class and probability of PD

def main():
    """Main function to run predictions."""
    parser = argparse.ArgumentParser(description='Predict Parkinson\'s disease from MRI scans')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input MRI scan (nifti format)')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save results')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Preprocess input
    input_tensor = preprocess_scan(args.input_file)
    
    # Run prediction
    class_idx, probability = predict(model, input_tensor, device)
    
    # Interpret results
    diagnosis = "Parkinson's Disease" if class_idx == 1 else "Control (No Parkinson's)"
    confidence = probability if class_idx == 1 else 1 - probability
    
    # Print results
    print("\n" + "="*50)
    print(f"PREDICTION RESULTS FOR: {os.path.basename(args.input_file)}")
    print("="*50)
    print(f"Diagnosis: {diagnosis}")
    print(f"Confidence: {confidence:.2%}")
    print("="*50)
    
    # Save results to file
    output_file = os.path.join(args.output_dir, f"{Path(args.input_file).stem}_prediction.txt")
    with open(output_file, 'w') as f:
        f.write(f"File: {args.input_file}\n")
        f.write(f"Diagnosis: {diagnosis}\n")
        f.write(f"Confidence: {confidence:.2%}\n")
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 