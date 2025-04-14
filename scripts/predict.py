#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prediction script for the Parkinson's MRI Detection model.

This script loads a trained model and makes predictions on new MRI scans.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from scipy.ndimage import zoom

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.transformers.region_attention_transformer import (
    create_region_attention_transformer,
    RegionAttentionTransformer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('pd_prediction')

# Define paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = ROOT_DIR / 'models' / 'pretrained'
VISUALIZATIONS_DIR = ROOT_DIR / 'visualizations'


def preprocess_mri(mri_path, target_size=(128, 128, 128)):
    """
    Preprocess an MRI scan for model input.
    
    Args:
        mri_path: Path to the MRI file
        target_size: Target size for the model
        
    Returns:
        Preprocessed volume as a tensor
    """
    # Load MRI volume
    nii_img = nib.load(mri_path)
    volume = nii_img.get_fdata()
    
    # Normalize intensity to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # Resize to target size if needed
    current_size = volume.shape
    if current_size != target_size:
        scale_factors = (
            target_size[0] / current_size[0],
            target_size[1] / current_size[1],
            target_size[2] / current_size[2]
        )
        volume = zoom(volume, scale_factors, order=1)
    
    # Convert to tensor and add batch and channel dimensions
    volume_tensor = torch.from_numpy(volume).float()
    volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    return volume_tensor


def visualize_prediction(mri_path, prediction, confidence, output_path):
    """
    Create a visualization of the prediction.
    
    Args:
        mri_path: Path to the MRI file
        prediction: Prediction label (0=Control, 1=PD)
        confidence: Prediction confidence (0-1)
        output_path: Path to save the visualization
    """
    # Load MRI volume
    nii_img = nib.load(mri_path)
    volume = nii_img.get_fdata()
    
    # Create a figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Prediction: {'Parkinson\\'s Disease' if prediction == 1 else 'Control'} "
        f"(Confidence: {confidence*100:.1f}%)",
        fontsize=16
    )
    
    # Get middle slices
    d, h, w = volume.shape
    slice_d = volume[d//2, :, :]
    slice_h = volume[:, h//2, :]
    slice_w = volume[:, :, w//2]
    
    # Plot slices
    axes[0, 0].imshow(slice_d, cmap='gray')
    axes[0, 0].set_title('Axial Slice')
    axes[0, 1].imshow(slice_h, cmap='gray')
    axes[0, 1].set_title('Coronal Slice')
    axes[0, 2].imshow(slice_w, cmap='gray')
    axes[0, 2].set_title('Sagittal Slice')
    
    # Plot confidence bar
    axes[1, 1].axis('off')
    axes[1, 0].axis('off')
    axes[1, 2].axis('off')
    
    # Create a confidence bar
    bar_x = np.linspace(0, 1, 100)
    bar_y = np.zeros_like(bar_x)
    
    axes[1, 1].barh(bar_y, bar_x, color='lightgray', height=0.5)
    axes[1, 1].barh(bar_y, [confidence], color='red' if prediction == 1 else 'green', height=0.5)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_title(f"Confidence: {confidence*100:.1f}%")
    
    # Add labels
    axes[1, 1].text(0.1, 0.2, "Control", fontsize=12)
    axes[1, 1].text(0.8, 0.2, "PD", fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_model(model_path, device):
    """
    Load a trained model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Create model with same architecture
    model = create_region_attention_transformer(
        input_size=(128, 128, 128),
        patch_size=16,
        embed_dim=256,
        depth=8,
        num_heads=8,
        num_classes=2,
        dropout=0.2,
        use_contrastive=True
    )
    
    # Load model weights with weights_only=False to handle PyTorch 2.6 security changes
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def predict(model, input_tensor, device):
    """
    Make a prediction on an input volume.
    
    Args:
        model: Trained model
        input_tensor: Input volume tensor
        device: Device to run inference on
        
    Returns:
        prediction: Prediction label (0=Control, 1=PD)
        confidence: Prediction confidence (0-1)
    """
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction and confidence
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()
        confidence = probabilities[0, prediction].item()
    
    return prediction, confidence


def main():
    """
    Main function for making predictions
    """
    parser = argparse.ArgumentParser(description='Make predictions with Parkinson\'s MRI Detection model')
    parser.add_argument('--mri_path', required=True, help='Path to the MRI scan to predict on')
    parser.add_argument('--model_path', default=None, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', default=None, help='Directory to save visualizations')
    # Add model architecture parameters
    parser.add_argument('--input_size', type=int, default=128, help='Input volume size')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    args = parser.parse_args()
    
    # Set up paths
    if args.model_path is None:
        # Find the latest model checkpoint if not specified
        model_files = list(MODELS_DIR.glob('best_model*.pth'))
        if not model_files:
            logger.error("No model checkpoints found. Please train a model first.")
            return
        args.model_path = str(max(model_files, key=os.path.getmtime))
    
    if args.output_dir is None:
        args.output_dir = str(VISUALIZATIONS_DIR)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"prediction_{Path(args.mri_path).stem}.png")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = create_region_attention_transformer(
        input_size=(args.input_size, args.input_size, args.input_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=2,
        dropout=0.2,
        use_contrastive=True
    )
    
    # Load model weights with weights_only=False to handle PyTorch 2.6 security changes
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Preprocess MRI
    logger.info(f"Preprocessing MRI from {args.mri_path}")
    input_tensor = preprocess_mri(args.mri_path)
    
    # Make prediction
    logger.info("Making prediction")
    prediction, confidence = predict(model, input_tensor, device)
    
    # Display result
    result = "Parkinson's Disease" if prediction == 1 else "Control (Healthy)"
    logger.info(f"Prediction: {result} (Confidence: {confidence*100:.1f}%)")
    
    # Create visualization
    logger.info(f"Creating visualization at {output_path}")
    visualize_prediction(args.mri_path, prediction, confidence, output_path)
    
    # Print instructions for clinical use
    logger.info("\nIMPORTANT: This prediction is for research purposes only.")
    logger.info("Clinical diagnosis of Parkinson's disease requires comprehensive evaluation by a neurologist.")


if __name__ == "__main__":
    main() 