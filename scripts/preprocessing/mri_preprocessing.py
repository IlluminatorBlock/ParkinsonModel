#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRI Preprocessing Pipeline for Parkinson's Disease Detection

This script performs preprocessing of MRI data for the Parkinson's Disease
detection project. It includes registration, skull stripping, intensity
normalization, and other preprocessing steps.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import time
import json
import shutil
from tqdm import tqdm
import logging
import glob
import pandas as pd

# Try to import optional dependencies
try:
    import ants
    ANTSPY_AVAILABLE = True
except ImportError:
    ANTSPY_AVAILABLE = False
    print("ANTsPy not available. Using SimpleITK for registration.")

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print("SimpleITK not available. Basic registration will be used.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)

logger = logging.getLogger('mri_preprocessing')

# Define paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
METADATA_DIR = ROOT_DIR / 'data' / 'metadata'

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Define preprocessing parameters
DEFAULT_PARAMS = {
    'registration': {
        'moving_type': 'T1',
        'fixed_image': 'MNI152_T1_1mm.nii.gz',
        'interpolation': 'linear',
        'transform_type': 'Rigid',
        'output_prefix': 'reg_',
    },
    'skull_stripping': {
        'method': 'bet',  # Options: bet, hd-bet, antspy
        'bet_f_value': 0.5,  # BET intensity threshold
        'output_prefix': 'skull_stripped_',
    },
    'intensity_normalization': {
        'method': 'z_score',  # Options: z_score, min_max, histogram_matching
        'mask_brain': True,
        'output_prefix': 'norm_',
    },
    'resampling': {
        'target_spacing': [1.0, 1.0, 1.0],  # in mm
        'interpolation': 'linear',
        'output_prefix': 'resampled_',
    },
    'bias_field_correction': {
        'method': 'n4',  # Options: n4
        'n4_shrink_factor': 4,
        'n4_iterations': [50, 50, 50, 50],
        'output_prefix': 'n4_',
    }
}

def register_mri_sitk(moving_image_path, fixed_image_path, output_path, params):
    """
    Register MRI image using SimpleITK
    """
    logger.info(f"Registering {moving_image_path} to {fixed_image_path} using SimpleITK")
    
    # Read images
    fixed_image = sitk.ReadImage(str(fixed_image_path), sitk.sitkFloat32)
    moving_image = sitk.ReadImage(str(moving_image_path), sitk.sitkFloat32)
    
    # Create registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Set similarity metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Set optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=100, 
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Setup for transformation
    if params['transform_type'].lower() == 'rigid':
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif params['transform_type'].lower() == 'affine':
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.AffineTransform(3), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    else:
        raise ValueError(f"Unsupported transform type: {params['transform_type']}")
        
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Execute registration
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    
    registered_image = resampler.Execute(moving_image)
    
    # Save registered image
    sitk.WriteImage(registered_image, str(output_path))
    logger.info(f"Saved registered image to {output_path}")
    
    return output_path

def register_mri_ants(moving_image_path, fixed_image_path, output_path, params):
    """
    Register MRI image using ANTsPy
    """
    logger.info(f"Registering {moving_image_path} to {fixed_image_path} using ANTsPy")
    
    # Read images
    fixed_image = ants.image_read(str(fixed_image_path))
    moving_image = ants.image_read(str(moving_image_path))
    
    # Perform registration
    transform_type = params['transform_type'].lower()
    if transform_type == 'rigid':
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='Rigid'
        )
    elif transform_type == 'affine':
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='Affine'
        )
    elif transform_type == 'syn':
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='SyN'
        )
    else:
        raise ValueError(f"Unsupported transform type: {transform_type}")
    
    # Get registered image
    registered_image = registration['warpedmovout']
    
    # Save registered image
    ants.image_write(registered_image, str(output_path))
    logger.info(f"Saved registered image to {output_path}")
    
    return output_path

def skull_strip_bet(input_image_path, output_path, params):
    """
    Perform skull stripping using FSL's BET
    This is implemented as a system call to FSL's bet command
    """
    logger.info(f"Skull stripping {input_image_path} using BET")
    
    # Check if FSL is installed
    if shutil.which('bet') is None:
        raise RuntimeError("FSL's BET command not found. Please install FSL.")
    
    # Build command
    cmd = f"bet {input_image_path} {output_path} -f {params['bet_f_value']} -g 0 -m"
    
    # Execute command
    logger.info(f"Running command: {cmd}")
    os.system(cmd)
    
    # Check if output file exists
    if not os.path.exists(output_path):
        raise RuntimeError(f"Skull stripping failed. Output file {output_path} does not exist.")
    
    logger.info(f"Saved skull-stripped image to {output_path}")
    return output_path

def skull_strip_sitk(input_image_path, output_path, params):
    """
    Perform simple skull stripping using SimpleITK threshold-based approach
    This is a basic implementation and not as effective as dedicated tools
    """
    logger.info(f"Skull stripping {input_image_path} using SimpleITK threshold method")
    
    # Read input image
    image = sitk.ReadImage(str(input_image_path), sitk.sitkFloat32)
    
    # Apply Otsu threshold to create a rough brain mask
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask = otsu_filter.Execute(image)
    
    # Apply morphological operations to clean up the mask
    close_filter = sitk.BinaryMorphologicalClosingImageFilter()
    close_filter.SetKernelRadius(3)
    mask = close_filter.Execute(mask)
    
    # Find largest connected component (the brain)
    connected_components = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(connected_components)
    brain_mask = relabeled == 1  # Select largest component
    
    # Apply mask to the original image
    skull_stripped = sitk.Mask(image, brain_mask)
    
    # Save output
    sitk.WriteImage(skull_stripped, str(output_path))
    
    # Also save the mask
    mask_path = str(output_path).replace('.nii', '_mask.nii')
    sitk.WriteImage(brain_mask, mask_path)
    
    logger.info(f"Saved skull-stripped image to {output_path}")
    logger.info(f"Saved brain mask to {mask_path}")
    
    return output_path

def normalize_intensity_z_score(input_image_path, output_path, mask_path=None, params=None):
    """
    Normalize image intensity using Z-score normalization
    """
    logger.info(f"Normalizing {input_image_path} using Z-score")
    
    # Load image
    image_obj = nib.load(str(input_image_path))
    image_data = image_obj.get_fdata()
    
    # Load mask if available
    if mask_path and os.path.exists(str(mask_path)):
        mask_obj = nib.load(str(mask_path))
        mask_data = mask_obj.get_fdata() > 0
    else:
        # Create a simple mask for non-zero voxels if no mask is provided
        mask_data = image_data > 0
    
    # Calculate mean and std of non-zero and masked voxels
    masked_data = image_data[mask_data]
    mean_val = np.mean(masked_data)
    std_val = np.std(masked_data)
    
    if std_val == 0:
        logger.warning("Standard deviation is zero. Cannot normalize with Z-score. Using original image.")
        normalized_data = image_data
    else:
        # Apply Z-score normalization
        normalized_data = np.zeros_like(image_data)
        normalized_data[mask_data] = (masked_data - mean_val) / std_val
    
    # Save normalized image
    normalized_img = nib.Nifti1Image(normalized_data, image_obj.affine, image_obj.header)
    nib.save(normalized_img, str(output_path))
    
    logger.info(f"Saved normalized image to {output_path}")
    return output_path

def normalize_intensity_min_max(input_image_path, output_path, mask_path=None, params=None):
    """
    Normalize image intensity using Min-Max scaling
    """
    logger.info(f"Normalizing {input_image_path} using Min-Max scaling")
    
    # Load image
    image_obj = nib.load(str(input_image_path))
    image_data = image_obj.get_fdata()
    
    # Load mask if available
    if mask_path and os.path.exists(str(mask_path)):
        mask_obj = nib.load(str(mask_path))
        mask_data = mask_obj.get_fdata() > 0
    else:
        # Create a simple mask for non-zero voxels if no mask is provided
        mask_data = image_data > 0
    
    # Calculate min and max of non-zero and masked voxels
    masked_data = image_data[mask_data]
    min_val = np.min(masked_data)
    max_val = np.max(masked_data)
    
    if max_val == min_val:
        logger.warning("Max value equals min value. Cannot normalize with Min-Max. Using original image.")
        normalized_data = image_data
    else:
        # Apply Min-Max normalization
        normalized_data = np.zeros_like(image_data)
        normalized_data[mask_data] = (masked_data - min_val) / (max_val - min_val)
    
    # Save normalized image
    normalized_img = nib.Nifti1Image(normalized_data, image_obj.affine, image_obj.header)
    nib.save(normalized_img, str(output_path))
    
    logger.info(f"Saved normalized image to {output_path}")
    return output_path

def resample_image(input_image_path, output_path, params):
    """
    Resample image to a target spacing
    """
    logger.info(f"Resampling {input_image_path} to spacing {params['target_spacing']}")
    
    if SITK_AVAILABLE:
        # Read image
        image = sitk.ReadImage(str(input_image_path))
        
        # Get current spacing
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # Calculate new size based on target spacing
        target_spacing = params['target_spacing']
        new_size = [
            int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
            int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
            int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
        ]
        
        # Set up resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        
        # Set interpolator
        if params['interpolation'].lower() == 'linear':
            resampler.SetInterpolator(sitk.sitkLinear)
        elif params['interpolation'].lower() == 'nearest':
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif params['interpolation'].lower() == 'bspline':
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            logger.warning(f"Unknown interpolation method: {params['interpolation']}. Using linear.")
            resampler.SetInterpolator(sitk.sitkLinear)
        
        # Execute resampling
        resampled_image = resampler.Execute(image)
        
        # Save resampled image
        sitk.WriteImage(resampled_image, str(output_path))
    else:
        # Fallback using nibabel (less accurate)
        image_obj = nib.load(str(input_image_path))
        image_data = image_obj.get_fdata()
        affine = image_obj.affine
        
        # Get current spacing (voxel size)
        current_spacing = np.array([
            np.sqrt(np.sum(affine[:3, 0] ** 2)),
            np.sqrt(np.sum(affine[:3, 1] ** 2)),
            np.sqrt(np.sum(affine[:3, 2] ** 2))
        ])
        
        # Calculate scaling factors
        target_spacing = np.array(params['target_spacing'])
        scaling_factors = current_spacing / target_spacing
        
        # Create new affine matrix with target spacing
        new_affine = affine.copy()
        for i in range(3):
            new_affine[:3, i] = affine[:3, i] / scaling_factors[i]
        
        # Save resampled image (just changing the affine, not interpolating data)
        # Note: This is a simplified approach that doesn't actually resample the data
        resampled_img = nib.Nifti1Image(image_data, new_affine)
        nib.save(resampled_img, str(output_path))
        logger.warning("Used simplified resampling with nibabel. For better results, install SimpleITK.")
    
    logger.info(f"Saved resampled image to {output_path}")
    return output_path

def apply_bias_field_correction_n4(input_image_path, output_path, mask_path=None, params=None):
    """
    Apply N4 bias field correction
    """
    logger.info(f"Applying N4 bias field correction to {input_image_path}")
    
    if not SITK_AVAILABLE:
        raise RuntimeError("SimpleITK is required for N4 bias field correction")
    
    # Read input image
    input_image = sitk.ReadImage(str(input_image_path), sitk.sitkFloat32)
    
    # Read mask if available
    mask = None
    if mask_path and os.path.exists(str(mask_path)):
        mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    
    # Set up bias field corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    
    # Set parameters
    corrector.SetMaximumNumberOfIterations(params['n4_iterations'])
    
    # Execute correction
    if mask is not None:
        corrected_image = corrector.Execute(input_image, mask)
    else:
        corrected_image = corrector.Execute(input_image)
    
    # Save corrected image
    sitk.WriteImage(corrected_image, str(output_path))
    
    logger.info(f"Saved bias-field corrected image to {output_path}")
    return output_path

def process_subject(subject_dir, output_dir, params):
    """
    Process a single subject's MRI data
    """
    subject_id = os.path.basename(subject_dir)
    logger.info(f"Processing subject: {subject_id}")
    
    # Create output directory for this subject
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # Find T1 MRI files
    t1_files = glob.glob(os.path.join(subject_dir, '**', '*T1*.nii*'), recursive=True)
    
    if not t1_files:
        logger.warning(f"No T1 MRI files found for subject {subject_id}")
        return None
    
    # Process each T1 file
    for t1_file in t1_files:
        # Extract session info from file path if available
        file_name = os.path.basename(t1_file)
        session_match = re.search(r'session[_-](\d+)', t1_file, re.IGNORECASE)
        session = f"session_{session_match.group(1)}" if session_match else "session_unknown"
        
        # Create session directory
        session_dir = os.path.join(subject_output_dir, session)
        os.makedirs(session_dir, exist_ok=True)
        
        # Copy original file
        original_copy = os.path.join(session_dir, f"original_{file_name}")
        shutil.copy2(t1_file, original_copy)
        
        current_file = original_copy
        
        # Apply preprocessing steps
        
        # 1. Registration
        if ANTSPY_AVAILABLE:
            fixed_image_path = os.path.join(ROOT_DIR, 'data', 'templates', params['registration']['fixed_image'])
            if os.path.exists(fixed_image_path):
                registered_file = os.path.join(session_dir, f"{params['registration']['output_prefix']}{file_name}")
                current_file = register_mri_ants(current_file, fixed_image_path, registered_file, params['registration'])
            else:
                logger.warning(f"Fixed image not found: {fixed_image_path}. Skipping registration.")
        elif SITK_AVAILABLE:
            fixed_image_path = os.path.join(ROOT_DIR, 'data', 'templates', params['registration']['fixed_image'])
            if os.path.exists(fixed_image_path):
                registered_file = os.path.join(session_dir, f"{params['registration']['output_prefix']}{file_name}")
                current_file = register_mri_sitk(current_file, fixed_image_path, registered_file, params['registration'])
            else:
                logger.warning(f"Fixed image not found: {fixed_image_path}. Skipping registration.")
        
        # 2. Skull stripping
        skull_stripped_file = os.path.join(session_dir, f"{params['skull_stripping']['output_prefix']}{file_name}")
        mask_file = os.path.join(session_dir, f"{params['skull_stripping']['output_prefix']}mask_{file_name}")
        
        if params['skull_stripping']['method'] == 'bet' and shutil.which('bet'):
            current_file = skull_strip_bet(current_file, skull_stripped_file, params['skull_stripping'])
        elif SITK_AVAILABLE:
            current_file = skull_strip_sitk(current_file, skull_stripped_file, params['skull_stripping'])
        else:
            logger.warning("No skull stripping method available. Skipping skull stripping.")
        
        # 3. Bias field correction
        if SITK_AVAILABLE:
            bias_corrected_file = os.path.join(session_dir, f"{params['bias_field_correction']['output_prefix']}{file_name}")
            current_file = apply_bias_field_correction_n4(current_file, bias_corrected_file, mask_file, params['bias_field_correction'])
        
        # 4. Intensity normalization
        if params['intensity_normalization']['method'] == 'z_score':
            normalized_file = os.path.join(session_dir, f"{params['intensity_normalization']['output_prefix']}{file_name}")
            current_file = normalize_intensity_z_score(current_file, normalized_file, mask_file, params['intensity_normalization'])
        elif params['intensity_normalization']['method'] == 'min_max':
            normalized_file = os.path.join(session_dir, f"{params['intensity_normalization']['output_prefix']}{file_name}")
            current_file = normalize_intensity_min_max(current_file, normalized_file, mask_file, params['intensity_normalization'])
        
        # 5. Resampling
        if SITK_AVAILABLE:
            resampled_file = os.path.join(session_dir, f"{params['resampling']['output_prefix']}{file_name}")
            current_file = resample_image(current_file, resampled_file, params['resampling'])
        
        # Log completion
        logger.info(f"Completed preprocessing for {t1_file}")
        
    return subject_output_dir

def main():
    """
    Main function for MRI preprocessing
    """
    parser = argparse.ArgumentParser(description='Preprocess MRI data for Parkinson\'s Disease Detection')
    parser.add_argument('--input_dir', type=str, default=str(RAW_DATA_DIR), 
                        help='Directory containing raw MRI data')
    parser.add_argument('--output_dir', type=str, default=str(PROCESSED_DATA_DIR), 
                        help='Directory to save preprocessed data')
    parser.add_argument('--subject', type=str, default=None,
                        help='Process specific subject (optional)')
    parser.add_argument('--params', type=str, default=None,
                        help='JSON file with preprocessing parameters (optional)')
    args = parser.parse_args()
    
    # Load parameters
    if args.params and os.path.exists(args.params):
        with open(args.params, 'r') as f:
            custom_params = json.load(f)
        params = DEFAULT_PARAMS.copy()
        # Update default params with custom ones
        for key, value in custom_params.items():
            if key in params:
                params[key].update(value)
            else:
                params[key] = value
    else:
        params = DEFAULT_PARAMS
    
    # Process subjects
    if args.subject:
        # Process specific subject
        subject_dir = os.path.join(args.input_dir, args.subject)
        if os.path.exists(subject_dir):
            process_subject(subject_dir, args.output_dir, params)
        else:
            logger.error(f"Subject directory not found: {subject_dir}")
    else:
        # Process all subjects
        subjects = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        logger.info(f"Found {len(subjects)} subjects")
        
        for subject in tqdm(subjects, desc="Processing subjects"):
            subject_dir = os.path.join(args.input_dir, subject)
            process_subject(subject_dir, args.output_dir, params)
    
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    # Add import for regex
    import re
    main() 