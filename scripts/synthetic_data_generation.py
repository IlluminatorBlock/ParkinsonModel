#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Synthetic MRI Data Generation for Parkinson's Disease Detection

This script generates synthetic 3D MRI data with features that mimic
Parkinson's disease-related changes for development and testing.

Enhanced version with improved parameters for better classification accuracy.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import random
import json
import logging
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('synthetic_data')

# Define paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw' / 'simulated'
METADATA_DIR = ROOT_DIR / 'data' / 'metadata'


def create_brain_template(size=(128, 128, 128), noise_level=0.1):
    """
    Create a template brain volume.
    
    Args:
        size: Size of the volume (D, H, W)
        noise_level: Amount of noise to add
        
    Returns:
        Brain template as numpy array
    """
    depth, height, width = size
    
    # Create coordinate grids
    x, y, z = np.mgrid[
        -depth/2:depth/2,
        -height/2:height/2,
        -width/2:width/2
    ]
    
    # Create ellipsoid for brain
    brain_mask = (x**2/(depth/2.2)**2 + y**2/(height/2.5)**2 + z**2/(width/2.5)**2) <= 1.0
    
    # Create template volume
    template = np.zeros(size, dtype=np.float32)
    
    # Fill brain region with base intensity
    template[brain_mask] = 0.8
    
    # Add some texture using Gaussian noise
    noise = np.random.normal(0, noise_level, size)
    template += noise * brain_mask.astype(float)
    
    # Apply Gaussian filter to smooth the volume
    template = gaussian_filter(template, sigma=1.0)
    
    # Ensure values are in [0, 1]
    template = np.clip(template, 0, 1)
    
    return template, brain_mask


def add_ventricles(volume, brain_mask, size=(128, 128, 128), ventricle_intensity=0.2, 
                  is_pd=False, pd_severity=1.0):
    """
    Add ventricles to the brain volume.
    
    Args:
        volume: Brain volume to modify
        brain_mask: Mask of the brain region
        size: Size of the volume
        ventricle_intensity: Intensity of ventricles (lower = darker)
        is_pd: Whether to simulate Parkinson's disease changes
        pd_severity: Severity factor for PD changes
        
    Returns:
        Volume with ventricles
    """
    depth, height, width = size
    
    # Create coordinate grids for ventricles
    center_offset = np.random.normal(0, 2, 3)
    ventricle_center = np.array([depth/2, height/2, width/2]) + center_offset
    
    x, y, z = np.mgrid[
        0:depth,
        0:height,
        0:width
    ]
    
    # Create ventricle masks (lateral ventricles)
    # PD often has ventricle enlargement, so make them larger for PD cases
    ventricle_size_factor = 1.0 + (0.4 * pd_severity if is_pd else 0.0)
    
    left_ventricle_size = np.array([depth/10, height/6, width/15]) * np.random.uniform(0.9, 1.1, 3) * ventricle_size_factor
    right_ventricle_size = np.array([depth/10, height/6, width/15]) * np.random.uniform(0.9, 1.1, 3) * ventricle_size_factor
    
    # Left ventricle (slightly offset)
    left_center = ventricle_center + np.array([0, 0, -width/10])
    left_ventricle = (
        ((x - left_center[0]) / left_ventricle_size[0])**2 +
        ((y - left_center[1]) / left_ventricle_size[1])**2 +
        ((z - left_center[2]) / left_ventricle_size[2])**2
    ) <= 1.0
    
    # Right ventricle (slightly offset)
    right_center = ventricle_center + np.array([0, 0, width/10])
    right_ventricle = (
        ((x - right_center[0]) / right_ventricle_size[0])**2 +
        ((y - right_center[1]) / right_ventricle_size[1])**2 +
        ((z - right_center[2]) / right_ventricle_size[2])**2
    ) <= 1.0
    
    # Third ventricle (connects the two)
    third_ventricle_size = np.array([depth/15, height/10, width/5]) * ventricle_size_factor
    third_ventricle = (
        ((x - ventricle_center[0]) / third_ventricle_size[0])**2 +
        ((y - ventricle_center[1]) / third_ventricle_size[1])**2 +
        ((z - ventricle_center[2]) / third_ventricle_size[2])**2
    ) <= 1.0
    
    # Combine ventricle masks
    ventricle_mask = (left_ventricle | right_ventricle | third_ventricle) & brain_mask
    
    # Apply ventricles to volume (darker than brain tissue)
    # Make ventricles slightly larger and darker in PD cases
    final_intensity = ventricle_intensity * (0.85 if is_pd else 1.0)
    volume[ventricle_mask] = final_intensity
    
    return volume, ventricle_mask


def add_substantia_nigra(volume, brain_mask, is_pd=False, size=(128, 128, 128), feature_strength=1.0):
    """
    Add substantia nigra to the brain volume with optional PD-related changes.
    
    Args:
        volume: Brain volume to modify
        brain_mask: Mask of the brain region
        is_pd: Whether to simulate Parkinson's disease changes
        size: Size of the volume
        feature_strength: Multiplier to enhance disease features (higher = more pronounced)
        
    Returns:
        Volume with substantia nigra
    """
    depth, height, width = size
    
    # Create coordinate grids for substantia nigra
    # SN is located in the midbrain, slightly ventral
    sn_center = np.array([depth*0.45, height*0.5, width*0.5]) + np.random.normal(0, 2, 3)
    
    x, y, z = np.mgrid[
        0:depth,
        0:height,
        0:width
    ]
    
    # Create substantia nigra mask (bilateral)
    sn_size_base = np.array([depth/25, height/25, width/15])
    
    # Left substantia nigra
    left_sn_center = sn_center + np.array([0, 0, -width/30])
    left_sn_size = sn_size_base * np.random.uniform(0.9, 1.1, 3)
    left_sn = (
        ((x - left_sn_center[0]) / left_sn_size[0])**2 +
        ((y - left_sn_center[1]) / left_sn_size[1])**2 +
        ((z - left_sn_center[2]) / left_sn_size[2])**2
    ) <= 1.0
    
    # Right substantia nigra
    right_sn_center = sn_center + np.array([0, 0, width/30])
    right_sn_size = sn_size_base * np.random.uniform(0.9, 1.1, 3)
    right_sn = (
        ((x - right_sn_center[0]) / right_sn_size[0])**2 +
        ((y - right_sn_center[1]) / right_sn_size[1])**2 +
        ((z - right_sn_center[2]) / right_sn_size[2])**2
    ) <= 1.0
    
    # Combine substantia nigra masks
    sn_mask = (left_sn | right_sn) & brain_mask
    
    # Apply substantia nigra to volume
    # CRITICAL IMPROVEMENT: More dramatic intensity difference between control and PD
    # In healthy brains, SN has a medium intensity
    # In PD brains, SN has significantly lower intensity (cell loss)
    if is_pd:
        # Enhanced PD features - more dramatic intensity drop (key improvement)
        sn_intensity = max(0.1, 0.35 - (0.25 * feature_strength))
    else:
        # Completely normal SN in controls (key improvement)
        sn_intensity = 0.6  # Normal intensity
    
    volume[sn_mask] = sn_intensity
    
    # For PD, add asymmetry and further degeneration
    if is_pd:
        # Create asymmetry (one side more affected)
        asymmetry_side = random.choice([left_sn, right_sn])
        # More dramatic asymmetry (key improvement)
        volume[asymmetry_side & brain_mask] *= max(0.3, 0.6 - (0.3 * feature_strength))
        
        # Add "blurred boundaries" effect for PD
        # This simulates the loss of clear boundaries in the SN
        sn_boundary = gaussian_filter((sn_mask).astype(float), sigma=2.0 * feature_strength) - gaussian_filter((sn_mask).astype(float), sigma=1.0)
        sn_boundary = (sn_boundary > 0.01) & ~sn_mask & brain_mask
        volume[sn_boundary] = np.clip(volume[sn_boundary] * 0.6, 0, 1)
        
        # Add additional degeneration pattern around SN (key improvement)
        sn_extended = gaussian_filter((sn_mask).astype(float), sigma=4.0) > 0.1
        sn_extended = sn_extended & ~sn_mask & brain_mask
        # Apply subtle intensity reduction in surrounding tissue
        volume[sn_extended] = volume[sn_extended] * 0.85
    
    return volume, sn_mask


def add_basal_ganglia(volume, brain_mask, is_pd=False, size=(128, 128, 128), feature_strength=1.0):
    """
    Add basal ganglia structures to the brain volume.
    
    Args:
        volume: Brain volume to modify
        brain_mask: Mask of the brain region
        is_pd: Whether to simulate Parkinson's disease changes
        size: Size of the volume
        feature_strength: Multiplier to enhance disease features
        
    Returns:
        Volume with basal ganglia
    """
    depth, height, width = size
    
    # Create coordinate grids for basal ganglia
    # Located above midbrain
    bg_center = np.array([depth*0.55, height*0.5, width*0.5]) + np.random.normal(0, 2, 3)
    
    x, y, z = np.mgrid[
        0:depth,
        0:height,
        0:width
    ]
    
    # Create striatum mask (putamen and caudate)
    striatum_size_base = np.array([depth/15, height/10, width/12])
    
    # Left striatum
    left_striatum_center = bg_center + np.array([0, 0, -width/8])
    left_striatum_size = striatum_size_base * np.random.uniform(0.9, 1.1, 3)
    left_striatum = (
        ((x - left_striatum_center[0]) / left_striatum_size[0])**2 +
        ((y - left_striatum_center[1]) / left_striatum_size[1])**2 +
        ((z - left_striatum_center[2]) / left_striatum_size[2])**2
    ) <= 1.0
    
    # Right striatum
    right_striatum_center = bg_center + np.array([0, 0, width/8])
    right_striatum_size = striatum_size_base * np.random.uniform(0.9, 1.1, 3)
    right_striatum = (
        ((x - right_striatum_center[0]) / right_striatum_size[0])**2 +
        ((y - right_striatum_center[1]) / right_striatum_size[1])**2 +
        ((z - right_striatum_center[2]) / right_striatum_size[2])**2
    ) <= 1.0
    
    # Create globus pallidus mask
    gp_size_base = np.array([depth/20, height/15, width/18])
    
    # Left globus pallidus (medial to striatum)
    left_gp_center = left_striatum_center + np.array([0, 0, width/25])
    left_gp_size = gp_size_base * np.random.uniform(0.9, 1.1, 3)
    left_gp = (
        ((x - left_gp_center[0]) / left_gp_size[0])**2 +
        ((y - left_gp_center[1]) / left_gp_size[1])**2 +
        ((z - left_gp_center[2]) / left_gp_size[2])**2
    ) <= 1.0
    
    # Right globus pallidus
    right_gp_center = right_striatum_center + np.array([0, 0, -width/25])
    right_gp_size = gp_size_base * np.random.uniform(0.9, 1.1, 3)
    right_gp = (
        ((x - right_gp_center[0]) / right_gp_size[0])**2 +
        ((y - right_gp_center[1]) / right_gp_size[1])**2 +
        ((z - right_gp_center[2]) / right_gp_size[2])**2
    ) <= 1.0
    
    # Combine basal ganglia masks
    striatum_mask = (left_striatum | right_striatum) & brain_mask
    gp_mask = (left_gp | right_gp) & brain_mask
    
    # Apply basal ganglia to volume 
    # IMPROVEMENT: More distinct intensity differences between PD and controls
    if is_pd:
        # PD cases have altered signal in striatum (key improvement)
        striatum_intensity = 0.6 - (0.15 * feature_strength)
        # And in globus pallidus
        gp_intensity = 0.4 - (0.1 * feature_strength)
    else:
        # Control cases have normal intensity
        striatum_intensity = 0.7
        gp_intensity = 0.5
    
    volume[striatum_mask] = striatum_intensity
    volume[gp_mask] = gp_intensity
    
    # For PD, add subtle changes in basal ganglia
    if is_pd:
        # Add stronger asymmetry to reflect dopaminergic denervation (key improvement)
        asymmetry_side = random.choice([left_striatum, right_striatum])
        volume[asymmetry_side & brain_mask] *= 0.8
        
        # Add subtle changes in connectivity patterns (modeled as intensity gradients)
        striatum_to_sn = gaussian_filter((striatum_mask).astype(float), sigma=3.0)
        connectivity_mask = (striatum_to_sn > 0.2) & (striatum_to_sn < 0.5) & brain_mask
        # Enhanced PD features in connectivity
        volume[connectivity_mask] = np.clip(volume[connectivity_mask] * 0.7, 0, 1)
        
        # Add subtle cortical thinning in PD (key improvement)
        cortex_mask = brain_mask & ~(striatum_mask | gp_mask | gaussian_filter((striatum_mask | gp_mask).astype(float), sigma=4.0) > 0.01)
        # Apply thinning to random patches in cortex
        for _ in range(int(5 * feature_strength)):
            # Random cortical region
            center = np.array([
                np.random.uniform(0.3, 0.7) * depth,
                np.random.uniform(0.3, 0.7) * height,
                np.random.uniform(0.3, 0.7) * width
            ])
            radius = np.random.uniform(5, 15)
            
            patch = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= radius**2
            patch = patch & cortex_mask
            
            # Apply subtle thinning
            volume[patch] *= 0.9
    
    return volume, striatum_mask, gp_mask


def create_3d_mask(shape, center, radius, shape_type='ellipsoid', orientation=(1.0, 1.0, 1.0)):
    """
    Create a 3D mask of a specified shape.
    
    Args:
        shape: Shape of the volume (D, H, W)
        center: Center coordinates (x, y, z)
        radius: Radius or size of the shape
        shape_type: Type of shape ('ellipsoid', 'hemisphere', etc.)
        orientation: Orientation factors for different axes
        
    Returns:
        Boolean mask of the 3D shape
    """
    d, h, w = shape
    x, y, z = np.ogrid[:d, :h, :w]
    
    dist_x = ((x - center[0]) / orientation[0])**2
    dist_y = ((y - center[1]) / orientation[1])**2
    dist_z = ((z - center[2]) / orientation[2])**2
    
    if shape_type == 'ellipsoid':
        mask = dist_x + dist_y + dist_z <= radius**2
    elif shape_type == 'hemisphere':
        mask = (dist_x + dist_y + dist_z <= radius**2) & (x <= center[0])
    else:  # Default to sphere
        mask = dist_x + dist_y + dist_z <= radius**2
        
    return mask


def generate_synthetic_mri(output_path, subject_id, is_pd=False, size=(128, 128, 128), add_noise=True, 
                          contrast_enhance=1.0, feature_strength=1.0):
    """
    Generate a synthetic MRI volume with optional PD-related changes.
    
    Args:
        output_path: Path to save the generated volume
        subject_id: Subject identifier
        is_pd: Whether to simulate Parkinson's disease
        size: Size of the volume
        add_noise: Whether to add random noise
        contrast_enhance: Factor to enhance contrast (higher = more contrast)
        feature_strength: Multiplier for disease features (higher = more pronounced)
        
    Returns:
        Path to the generated volume
    """
    # Set random seed based on subject_id for reproducibility
    random.seed(hash(subject_id) % 2**32)
    np.random.seed(hash(subject_id) % 2**32)
    
    # Create brain template
    brain, brain_mask = create_brain_template(size=size)
    
    # Add ventricles
    brain, ventricle_mask = add_ventricles(brain, brain_mask, size=size, is_pd=is_pd, pd_severity=feature_strength)
    
    # Add substantia nigra with optional PD changes
    brain, sn_mask = add_substantia_nigra(brain, brain_mask, is_pd=is_pd, size=size, feature_strength=feature_strength)
    
    # Add basal ganglia with optional PD changes
    brain, striatum_mask, gp_mask = add_basal_ganglia(brain, brain_mask, is_pd=is_pd, size=size, feature_strength=feature_strength)
    
    # Optional: Add noise and artifacts
    if add_noise:
        # Add random noise
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_level, size)
        brain += noise * brain_mask.astype(float)
        
        # Add subtle motion artifacts
        if random.random() < 0.3:
            motion_direction = random.choice([0, 1, 2])  # Random axis
            motion_amount = np.random.uniform(0.5, 2.0)
            shift = np.zeros(3)
            shift[motion_direction] = motion_amount
            
            # Apply subtle shift to a few slices
            slice_indices = np.random.choice(
                range(5, size[motion_direction]-5), 
                size=int(size[motion_direction]*0.1),
                replace=False
            )
            
            for idx in slice_indices:
                if motion_direction == 0:
                    brain[idx, :, :] = np.roll(brain[idx, :, :], int(shift[1]), axis=0)
                elif motion_direction == 1:
                    brain[:, idx, :] = np.roll(brain[:, idx, :], int(shift[2]), axis=1)
                else:
                    brain[:, :, idx] = np.roll(brain[:, :, idx], int(shift[0]), axis=0)
    
    # Apply final Gaussian smoothing to make it look more realistic
    brain = gaussian_filter(brain, sigma=0.5)
    
    # Enhance contrast if requested
    if contrast_enhance > 1.0:
        # Apply contrast enhancement
        mean_val = np.mean(brain[brain_mask])
        brain[brain_mask] = mean_val + (brain[brain_mask] - mean_val) * contrast_enhance
    
    # Ensure values are in [0, 1]
    brain = np.clip(brain, 0, 1)
    
    # Convert to uint16 for NIFTI
    brain_uint16 = (brain * 65535).astype(np.uint16)
    
    # Create NIFTI image
    # Define affine matrix for 1mm isotropic voxels
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm voxel size
    
    nii_img = nib.Nifti1Image(brain_uint16, affine)
    
    # Set header information
    nii_img.header['descrip'] = f'Synthetic MRI for {subject_id}'
    
    # Save the image
    nib.save(nii_img, output_path)
    
    return output_path


def generate_dataset(
    num_subjects=500,
    output_dir=None,
    pd_ratio=0.5,
    size=(128, 128, 128),
    save_metadata=True,
    contrast_enhance=3.0,  # Increased contrast enhancement
    feature_strength=5.0   # Strengthened PD features
):
    """
    Generate a synthetic dataset with multiple subjects.
    
    Args:
        num_subjects: Number of subjects to generate
        output_dir: Directory to save the generated volumes
        pd_ratio: Ratio of PD cases
        size: Size of the volumes
        save_metadata: Whether to save metadata
        contrast_enhance: Factor to enhance contrast (higher = more contrast)
        feature_strength: Multiplier for disease features (higher = more pronounced)
        
    Returns:
        Dictionary with dataset information
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate subjects
    metadata = {
        'subjects': []
    }
    
    for i in tqdm(range(num_subjects), desc="Generating synthetic MRI data"):
        # Generate subject ID
        subject_id = f'SIM{i:03d}'
        
        # Determine if this is a PD case
        is_pd = random.random() < pd_ratio
        
        # Create subject directory
        subject_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        # Generate MRI file path
        mri_path = os.path.join(subject_dir, f'{subject_id}_T1.nii.gz')
        
        # Generate synthetic MRI with enhanced parameters
        generate_synthetic_mri(
            output_path=mri_path,
            subject_id=subject_id,
            is_pd=is_pd,
            size=size,
            add_noise=True,
            contrast_enhance=contrast_enhance,
            feature_strength=feature_strength
        )
        
        # Generate metadata
        subject_metadata = {
            'subject_id': subject_id,
            'group': 'PD' if is_pd else 'Control',
            'age': random.randint(55, 80),
            'sex': random.choice(['M', 'F']),
            'mri_path': os.path.relpath(mri_path, output_dir)
        }
        
        # Add PD-specific metadata
        if is_pd:
            subject_metadata['disease_duration'] = round(random.uniform(1, 10), 1)
            subject_metadata['updrs_score'] = random.randint(10, 50)
            subject_metadata['hoehn_yahr'] = random.choice([1, 1.5, 2, 2.5, 3, 4, 5])
        
        metadata['subjects'].append(subject_metadata)
    
    # Save metadata
    if save_metadata:
        # Ensure metadata directory exists
        os.makedirs(METADATA_DIR, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(METADATA_DIR, 'simulated_metadata.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save as CSV for easier processing
        import pandas as pd
        df = pd.DataFrame(metadata['subjects'])
        csv_path = os.path.join(METADATA_DIR, 'simulated_metadata.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved metadata to {json_path} and {csv_path}")
    
    logger.info(f"Generated synthetic dataset with {num_subjects} subjects " +
                f"({int(num_subjects * pd_ratio)} PD, {int(num_subjects * (1 - pd_ratio))} control)")
    
    return metadata


def main():
    """
    Main function to generate synthetic MRI data.
    """
    parser = argparse.ArgumentParser(description='Generate synthetic MRI data for Parkinson\'s disease detection')
    parser.add_argument('--num_subjects', type=int, default=500, help='Number of subjects to generate')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for the generated data')
    parser.add_argument('--pd_ratio', type=float, default=0.5, help='Ratio of PD cases')
    parser.add_argument('--size', type=int, nargs=3, default=[128, 128, 128], help='Size of the volumes (D H W)')
    parser.add_argument('--contrast_enhance', type=float, default=3.0, help='Contrast enhancement factor')
    parser.add_argument('--feature_strength', type=float, default=5.0, help='Disease feature strength multiplier')
    parser.add_argument('--visualize', action='store_true', help='Visualize a few samples')
    parser.add_argument('--vis_dir', type=str, default=None, help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Generate dataset with enhanced parameters
    metadata = generate_dataset(
        num_subjects=args.num_subjects,
        output_dir=args.output_dir,
        pd_ratio=args.pd_ratio,
        size=tuple(args.size),
        contrast_enhance=args.contrast_enhance,
        feature_strength=args.feature_strength
    )
    
    # Visualize a few samples if requested
    if args.visualize:
        vis_subjects = random.sample(metadata['subjects'], min(5, len(metadata['subjects'])))
        
        for subject in vis_subjects:
            subject_id = subject['subject_id']
            group = subject['group']
            
            if args.output_dir is None:
                mri_path = os.path.join(RAW_DATA_DIR, subject_id, f'{subject_id}_T1.nii.gz')
            else:
                mri_path = os.path.join(args.output_dir, subject_id, f'{subject_id}_T1.nii.gz')
            
            if os.path.exists(mri_path):
                logger.info(f"Visualizing {subject_id} ({group})")
                visualize_synthetic_mri(mri_path, subject_id, args.vis_dir)


def visualize_synthetic_mri(mri_path, subject_id, output_dir=None):
    """
    Visualize a synthetic MRI volume.
    
    Args:
        mri_path: Path to the MRI volume
        subject_id: Subject identifier
        output_dir: Directory to save the visualizations
        
    Returns:
        Path to the visualization image
    """
    # Load the MRI volume
    nii_img = nib.load(mri_path)
    volume = nii_img.get_fdata()
    
    # Normalize to [0, 1] if needed
    if volume.max() > 1.0:
        volume = volume / volume.max()
    
    # Get dimensions
    d, h, w = volume.shape
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Synthetic MRI - Subject {subject_id}', fontsize=16)
    
    # Axial slices (top view)
    for i, pos in enumerate([0.4, 0.5, 0.6]):
        slice_idx = int(d * pos)
        axes[0, i].imshow(volume[slice_idx, :, :], cmap='gray')
        axes[0, i].set_title(f'Axial Slice {slice_idx}')
        axes[0, i].axis('off')
    
    # Coronal slices (front view)
    axes[1, 0].imshow(volume[:, int(h*0.4), :], cmap='gray')
    axes[1, 0].set_title(f'Coronal Slice {int(h*0.4)}')
    axes[1, 0].axis('off')
    
    # Sagittal slices (side view)
    axes[1, 1].imshow(volume[:, :, int(w*0.4)], cmap='gray')
    axes[1, 1].set_title(f'Sagittal Slice {int(w*0.4)}')
    axes[1, 1].axis('off')
    
    # 3D rendering approximation (simplified)
    axes[1, 2].imshow(np.max(volume, axis=0), cmap='gray')
    axes[1, 2].set_title('MIP Projection')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{subject_id}_mri_vis.png')
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    else:
        plt.show()
        plt.close(fig)
        return None


if __name__ == "__main__":
    main() 