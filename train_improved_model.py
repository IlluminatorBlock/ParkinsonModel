#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to train an improved model for Parkinson's disease detection,
with better synthetic data generation and training parameters.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Train an improved model for Parkinson's disease detection")
    parser.add_argument("--subjects", type=int, default=1000, help="Number of subjects to generate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--model_type", type=str, default="deep_voxel_3d", 
                        choices=["region_attention", "deep_voxel_3d", "efficientnet3d"], 
                        help="Model architecture to use")
    parser.add_argument("--output_dir", type=str, default="models/improved", 
                        help="Directory to save the improved model")
    return parser.parse_args()

def run_command(command):
    """Run a command and print its output"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Stream the output
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    
    # Wait for the process to finish and get the return code
    return_code = process.wait()
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
        sys.exit(return_code)

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("data/raw/improved", exist_ok=True)
    os.makedirs("data/processed/improved", exist_ok=True)
    
    # Save configuration
    config = {
        "timestamp": timestamp,
        "subjects": args.subjects,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model_type": args.model_type,
        "output_dir": args.output_dir
    }
    
    with open(os.path.join(args.output_dir, f"config_{timestamp}.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    print("=" * 80)
    print(f"STARTING IMPROVED TRAINING PIPELINE - {timestamp}")
    print("=" * 80)
    
    # Step 1: Generate improved synthetic data with increased parameters
    print("\n[Step 1] Generating improved synthetic data...")
    run_command(f"python scripts/synthetic_data_generation.py "
                f"--output_dir data/raw/improved "
                f"--num_subjects {args.subjects} "
                f"--pd_ratio 0.5 "
                f"--contrast_enhance 5.5 "  # Increase from 4.0
                f"--feature_strength 8.0 "  # Increase from 6.5
                f"--sn_intensity_pd 0.15 "  # Explicitly set lower SN intensity for PD
                f"--asymmetry_factor 0.7 "  # Greater asymmetry (more realistic)
                f"--add_dopamine_pathways true") # Model nigro-striatal connections
    
    # Step 2: Enhanced preprocessing with more sophisticated augmentation
    print("\n[Step 2] Preprocessing data with advanced techniques...")
    run_command(f"python scripts/preprocess_data.py "
                f"--input_dir data/raw/improved "
                f"--output_dir data/processed/improved "
                f"--normalize z-score "  # Better normalization technique
                f"--augment strong "
                f"--histogram_equalization "  # Add histogram equalization
                f"--noise_reduction ")  # Add noise reduction
    
    # Step 3: Train an improved model with optimal hyperparameters
    print("\n[Step 3] Training improved model with optimized settings...")
    # Note the class weights [2.5, 1.0] - stronger penalty for false positives
    run_command(f"python training/train.py "
                f"--data_dir data/processed/improved "
                f"--model_type {args.model_type} "
                f"--batch_size {args.batch_size} "
                f"--epochs {args.epochs} "
                f"--learning_rate 0.0001 "
                f"--min_learning_rate 1e-7 "  # Prevent learning rate from getting too small
                f"--dropout 0.4 "  # Increased dropout for better generalization
                f"--class_weights 2.5 1.0 "  # Stronger penalty for false positives
                f"--augmentation strong "
                f"--scheduler cosine "
                f"--mixup_alpha 0.4 "  # Enable mixup augmentation
                f"--label_smoothing 0.1 "  # Add label smoothing for better generalization
                f"--patience 15 "  # More patience for early stopping
                f"--warmup_epochs 5 "  # Add warmup for more stable training
                f"--save_dir {args.output_dir} "
                f"--model_suffix _improved_{timestamp}")
    
    # Step 4: Evaluate the model with comprehensive metrics
    print("\n[Step 4] Evaluating improved model with detailed metrics...")
    run_command(f"python check_model.py "
                f"--model_path {args.output_dir}/best_model_improved_{timestamp}.pt "
                f"--test_data data/processed/improved/test "
                f"--output_dir visualizations/improved_{timestamp} "
                f"--detailed_metrics "  # Generate more detailed metrics
                f"--confusion_matrix "  # Create confusion matrix visualization
                f"--roc_curve ")  # Generate ROC curve
    
    print("\n" + "=" * 80)
    print(f"TRAINING PIPELINE COMPLETED - {timestamp}")
    print(f"Model saved to: {args.output_dir}/best_model_improved_{timestamp}.pt")
    print(f"Evaluation results: visualizations/improved_{timestamp}/")
    print("=" * 80)

if __name__ == "__main__":
    main() 