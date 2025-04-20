#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessing script for Parkinson's MRI data.
This script handles processing of MRI data for training the Parkinson's detection model.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import logging
import json
import random
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocessing.mri_preprocessing import normalize_volume, augment_volume

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pd_preprocessing')

# Define constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
METADATA_DIR = os.path.join(ROOT_DIR, 'data', 'metadata')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Preprocess MRI data for Parkinson's detection")
    parser.add_argument('--input_dir', type=str, default=os.path.join(RAW_DATA_DIR, 'improved'),
                        help='Directory containing raw MRI data')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROCESSED_DATA_DIR, 'improved'),
                        help='Directory to save processed data')
    parser.add_argument('--metadata', type=str, 
                        default=os.path.join(METADATA_DIR, 'simulated_metadata.csv'),
                        help='Path to metadata file')
    parser.add_argument('--split_ratio', type=float, default=0.2,
                        help='Test split ratio')
    parser.add_argument('--normalize', type=str, default='z-score',
                        choices=['z-score', 'min-max', 'percentile'],
                        help='Normalization method')
    parser.add_argument('--augment', type=str, default='strong',
                        choices=['none', 'basic', 'strong'],
                        help='Augmentation level')
    parser.add_argument('--histogram_equalization', action='store_true',
                        help='Apply histogram equalization')
    parser.add_argument('--noise_reduction', action='store_true',
                        help='Apply noise reduction')
    parser.add_argument('--resample', type=bool, default=True,
                        help='Resample to isotropic resolution')
    parser.add_argument('--resize', type=int, default=128,
                        help='Resize dimension (resulting volume will be resize×resize×resize)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def apply_histogram_equalization(volume, bins=256):
    """
    Apply histogram equalization to MRI volume
    
    Args:
        volume: 3D MRI volume
        bins: Number of bins for histogram
        
    Returns:
        Histogram equalized volume
    """
    # Get the minimum and maximum values
    v_min, v_max = volume.min(), volume.max()
    
    # Calculate histogram
    hist, bin_edges = np.histogram(volume, bins=bins, range=(v_min, v_max))
    
    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    
    # Normalize the CDF
    cdf = cdf / float(cdf[-1])
    
    # Linear interpolation of CDF to map input values to equalized values
    equalized_volume = np.interp(volume.flatten(), bin_edges[:-1], cdf)
    
    # Reshape back to original shape
    equalized_volume = equalized_volume.reshape(volume.shape)
    
    return equalized_volume

def apply_noise_reduction(volume, sigma=0.5):
    """
    Apply Gaussian filtering for noise reduction
    
    Args:
        volume: 3D MRI volume
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Noise-reduced volume
    """
    return gaussian_filter(volume, sigma=sigma)

def normalize_volume_enhanced(volume, method='z-score'):
    """
    Normalize volume with enhanced options
    
    Args:
        volume: 3D MRI volume
        method: Normalization method ('z-score', 'min-max', 'percentile')
        
    Returns:
        Normalized volume
    """
    # Create a mask of non-zero voxels (brain region)
    mask = volume > 0
    
    if method == 'z-score':
        # Z-score normalization on brain region
        mean = np.mean(volume[mask])
        std = np.std(volume[mask])
        if std > 0:
            normalized = np.zeros_like(volume, dtype=np.float32)
            normalized[mask] = (volume[mask] - mean) / std
        else:
            normalized = volume - mean
    
    elif method == 'min-max':
        # Min-max normalization to [0, 1]
        min_val = np.min(volume[mask])
        max_val = np.max(volume[mask])
        if max_val > min_val:
            normalized = np.zeros_like(volume, dtype=np.float32)
            normalized[mask] = (volume[mask] - min_val) / (max_val - min_val)
        else:
            normalized = volume
    
    elif method == 'percentile':
        # Percentile-based normalization (robust to outliers)
        p1, p99 = np.percentile(volume[mask], (1, 99))
        normalized = np.zeros_like(volume, dtype=np.float32)
        normalized[mask] = np.clip((volume[mask] - p1) / (p99 - p1), 0, 1)
    
    else:
        normalized = volume
    
    return normalized

def augment_volume_enhanced(volume, augmentation_level='strong', is_pd=None):
    """
    Apply enhanced augmentation to MRI volume
    
    Args:
        volume: 3D MRI volume
        augmentation_level: Level of augmentation ('none', 'basic', 'strong')
        is_pd: Whether this is a PD subject (for targeted augmentation)
        
    Returns:
        Augmented volume
    """
    if augmentation_level == 'none':
        return volume
    
    # Create a copy to avoid modifying the original
    augmented = volume.copy()
    
    # Basic augmentations
    if random.random() < 0.5:
        # Random flip along one axis
        flip_axis = random.randint(0, 2)
        augmented = np.flip(augmented, axis=flip_axis)
    
    if augmentation_level == 'strong':
        # Strong augmentations
        
        # Random rotation (small angles)
        if random.random() < 0.7:
            angle = random.uniform(-10, 10)
            augmented = ndimage.rotate(augmented, angle, axes=(0, 1), reshape=False, order=1)
        
        # Random shift (small)
        if random.random() < 0.5:
            shift = [random.uniform(-3, 3) for _ in range(3)]
            augmented = ndimage.shift(augmented, shift, order=1)
        
        # Random elastic deformation (subtle)
        if random.random() < 0.3:
            shape = augmented.shape
            # Create random displacement fields
            displacement = np.random.randn(3, *shape) * 2
            # Smooth the displacement fields
            for i in range(3):
                displacement[i] = gaussian_filter(displacement[i], sigma=10)
            
            # Create meshgrid
            points = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij'
            )
            
            # Add displacement
            points = [p + d for p, d in zip(points, displacement)]
            
            # Interpolate
            coordinates = np.stack(points, axis=-1)
            augmented = ndimage.map_coordinates(augmented, coordinates.transpose(3, 0, 1, 2), order=1)
        
        # Random intensity variations
        if random.random() < 0.5:
            # Global intensity change
            intensity_factor = random.uniform(0.9, 1.1)
            augmented = augmented * intensity_factor
        
        # Targeted augmentation for PD features if known
        if is_pd is not None:
            # For PD cases, slightly enhance substantia nigra changes
            if is_pd and random.random() < 0.5:
                # This is simplified - in reality, would need anatomical knowledge
                # Just simulating with a small central region modification
                center = np.array(augmented.shape) // 2
                radius = min(augmented.shape) // 10
                
                # Create a small mask in the midbrain region (simplified)
                mask = np.zeros_like(augmented, dtype=bool)
                x, y, z = np.ogrid[:augmented.shape[0], :augmented.shape[1], :augmented.shape[2]]
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                mask[dist <= radius] = True
                
                # Apply subtle intensity change
                factor = random.uniform(0.85, 0.95)
                augmented[mask] = augmented[mask] * factor
    
    return augmented

def preprocess_mri(
    input_path, 
    output_path, 
    normalize_method='z-score',
    augment_level='strong',
    apply_hist_eq=False,
    apply_noise_red=False,
    resize_dim=128,
    is_pd=None
):
    """
    Preprocess a single MRI volume
    
    Args:
        input_path: Path to input MRI
        output_path: Path to save processed MRI
        normalize_method: Method for normalization
        augment_level: Level of augmentation
        apply_hist_eq: Whether to apply histogram equalization
        apply_noise_red: Whether to apply noise reduction
        resize_dim: Dimension to resize to
        is_pd: Whether this is a PD case (for targeted augmentation)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load MRI
        nii_img = nib.load(input_path)
        volume = nii_img.get_fdata()
        
        # Apply noise reduction if requested
        if apply_noise_red:
            volume = apply_noise_reduction(volume)
        
        # Apply histogram equalization if requested
        if apply_hist_eq:
            volume = apply_histogram_equalization(volume)
        
        # Normalize volume
        volume = normalize_volume_enhanced(volume, method=normalize_method)
        
        # Resize if necessary
        if resize_dim and (volume.shape[0] != resize_dim or 
                           volume.shape[1] != resize_dim or 
                           volume.shape[2] != resize_dim):
            # Resample to isotropic resolution
            zoom_factors = [resize_dim / s for s in volume.shape]
            volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        # Apply augmentation
        volume = augment_volume_enhanced(volume, augment_level, is_pd)
        
        # Save as NIfTI
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_nii = nib.Nifti1Image(volume, nii_img.affine)
        nib.save(new_nii, output_path)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    """Main function to preprocess all MRI data"""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    test_dir = os.path.join(args.output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    logger.info(f"Starting preprocessing with normalize={args.normalize}, augment={args.augment}")
    logger.info(f"Histogram equalization: {args.histogram_equalization}, Noise reduction: {args.noise_reduction}")
    
    # Get list of subjects from input directory
    subject_dirs = []
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path) and item.startswith('SIM'):
            subject_dirs.append(item)
    
    if not subject_dirs:
        logger.error(f"No subject directories found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(subject_dirs)} subjects")
    
    # Load metadata to determine PD status if available
    metadata = {}
    if os.path.exists(args.metadata):
        try:
            import pandas as pd
            meta_df = pd.read_csv(args.metadata)
            for _, row in meta_df.iterrows():
                subject_id = row['subject_id']
                is_pd = row['group'] == 'PD'
                metadata[subject_id] = {'is_pd': is_pd}
            logger.info(f"Loaded metadata for {len(metadata)} subjects")
        except Exception as e:
            logger.warning(f"Error loading metadata: {str(e)}")
    
    # Split subjects into train, validation, and test sets
    train_val_subjects, test_subjects = train_test_split(
        subject_dirs, test_size=args.split_ratio, random_state=args.random_seed
    )
    
    # Further split train into train and validation
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=0.2, random_state=args.random_seed
    )
    
    logger.info(f"Split: {len(train_subjects)} train, {len(val_subjects)} validation, {len(test_subjects)} test")
    
    # Save split information
    split_info = {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects
    }
    
    with open(os.path.join(args.output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Process train set (with augmentation)
    logger.info("Processing training set...")
    for subject_id in tqdm(train_subjects):
        # Get MRI file
        mri_file = None
        subject_dir = os.path.join(args.input_dir, subject_id)
        for file in os.listdir(subject_dir):
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                mri_file = os.path.join(subject_dir, file)
                break
        
        if not mri_file:
            logger.warning(f"No MRI file found for subject {subject_id}")
            continue
        
        # Get PD status from metadata if available
        is_pd = metadata.get(subject_id, {}).get('is_pd', None)
        
        # Set output path
        output_file = os.path.join(train_dir, f"{subject_id}.nii.gz")
        
        # Process MRI
        preprocess_mri(
            mri_file, 
            output_file, 
            normalize_method=args.normalize,
            augment_level=args.augment,
            apply_hist_eq=args.histogram_equalization,
            apply_noise_red=args.noise_reduction,
            resize_dim=args.resize,
            is_pd=is_pd
        )
    
    # Process validation set (no augmentation)
    logger.info("Processing validation set...")
    for subject_id in tqdm(val_subjects):
        mri_file = None
        subject_dir = os.path.join(args.input_dir, subject_id)
        for file in os.listdir(subject_dir):
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                mri_file = os.path.join(subject_dir, file)
                break
        
        if not mri_file:
            continue
        
        output_file = os.path.join(val_dir, f"{subject_id}.nii.gz")
        
        # Process without augmentation
        preprocess_mri(
            mri_file, 
            output_file, 
            normalize_method=args.normalize,
            augment_level='none',  # No augmentation for validation
            apply_hist_eq=args.histogram_equalization,
            apply_noise_red=args.noise_reduction,
            resize_dim=args.resize,
            is_pd=None
        )
    
    # Process test set (no augmentation)
    logger.info("Processing test set...")
    for subject_id in tqdm(test_subjects):
        mri_file = None
        subject_dir = os.path.join(args.input_dir, subject_id)
        for file in os.listdir(subject_dir):
            if file.endswith('.nii.gz') or file.endswith('.nii'):
                mri_file = os.path.join(subject_dir, file)
                break
        
        if not mri_file:
            continue
        
        output_file = os.path.join(test_dir, f"{subject_id}.nii.gz")
        
        # Process without augmentation
        preprocess_mri(
            mri_file, 
            output_file, 
            normalize_method=args.normalize,
            augment_level='none',  # No augmentation for test
            apply_hist_eq=args.histogram_equalization,
            apply_noise_red=args.noise_reduction,
            resize_dim=args.resize,
            is_pd=None
        )
    
    # Create metadata files for each set
    for split, subjects in zip(['train', 'val', 'test'], [train_subjects, val_subjects, test_subjects]):
        split_metadata = []
        for subject_id in subjects:
            is_pd = metadata.get(subject_id, {}).get('is_pd', False)
            split_metadata.append({
                'subject_id': subject_id,
                'group': 'PD' if is_pd else 'Control',
                'file_path': f"{subject_id}.nii.gz"
            })
        
        # Save metadata
        import pandas as pd
        df = pd.DataFrame(split_metadata)
        df.to_csv(os.path.join(args.output_dir, f"{split}_metadata.csv"), index=False)
    
    logger.info("Preprocessing completed!")

if __name__ == "__main__":
    main() 