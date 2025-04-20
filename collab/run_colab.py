#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Colab Runner for Parkinson's Disease Detection

This script automates the full training pipeline for the Parkinson's disease
detection model in Google Colab. It handles setup, data generation, training,
and saving results to Google Drive.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('colab_run.log')
    ]
)

logger = logging.getLogger('pd_colab_runner')

def run_command(command, desc=None):
    """Run a command and stream its output"""
    if desc:
        logger.info(f"{desc}")
    
    logger.info(f"Running command: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Stream the output
    for line in process.stdout:
        logger.info(line.strip())
    
    # Get the return code
    return_code = process.wait()
    if return_code != 0:
        logger.error(f"Command failed with return code {return_code}")
    
    return return_code == 0

def setup_colab_env():
    """Setup the Colab environment"""
    logger.info("Setting up Colab environment")
    
    # Create directories
    for directory in ["data", "data/raw", "data/raw/improved", "data/metadata", 
                      "models", "models/improved", "visualizations", "training",
                      "training/results"]:
        os.makedirs(directory, exist_ok=True)
    
    # Try to mount Google Drive
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive"):
            logger.info("Mounting Google Drive")
            drive.mount('/content/drive')
        else:
            logger.info("Google Drive already mounted")
    except ImportError:
        logger.warning("Not running in Google Colab, skipping Drive mount")
    
    # Install required packages
    logger.info("Installing required packages")
    packages = [
        "nibabel", 
        "torch", 
        "torchvision", 
        "tqdm", 
        "matplotlib", 
        "scikit-learn", 
        "scipy"
    ]
    
    return run_command(f"pip install {' '.join(packages)}", "Installing packages")

def clone_repo(repo_url, branch="main"):
    """Clone the repository if needed"""
    if os.path.exists(".git"):
        logger.info("Repository already cloned, pulling latest changes")
        return run_command(f"git pull origin {branch}", "Pulling latest changes")
    else:
        logger.info(f"Cloning repository from {repo_url}")
        return run_command(f"git clone {repo_url} .", "Cloning repository")

def run_training_pipeline(args):
    """Run the complete training pipeline"""
    logger.info("Starting the Parkinson's Disease Detection training pipeline")
    
    # Generate synthetic data
    data_cmd = [
        "python", 
        "pd_synthetic_data.py",
        "--num_subjects", str(args.num_subjects),
        "--output_dir", "data/raw/improved",
        "--pd_ratio", "0.5",
        "--feature_strength", "8.0",
        "--contrast_enhance", "6.0"
    ]
    
    if args.visualize:
        data_cmd.append("--visualize")
    
    data_success = run_command(" ".join(data_cmd), "Generating synthetic data")
    if not data_success:
        logger.error("Data generation failed")
        return False
    
    # Train model
    train_cmd = [
        "python",
        "pd_model_trainer.py",
        "--data_dir", "data/raw/improved",
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--device", "cuda" if args.use_gpu else "cpu",
        "--save_to_drive"
    ]
    
    if args.mixed_precision:
        train_cmd.append("--mixed_precision")
    
    train_success = run_command(" ".join(train_cmd), "Training model")
    if not train_success:
        logger.error("Model training failed")
        return False
    
    logger.info("Training pipeline completed successfully")
    return True

def main():
    """Parse arguments and run the Colab pipeline"""
    parser = argparse.ArgumentParser(description="Run Parkinson's Disease Detection in Colab")
    
    # Data generation parameters
    parser.add_argument("--num_subjects", type=int, default=1000,
                       help="Number of subjects to generate")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations of synthetic data")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="Use GPU for training if available")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="Use mixed precision training if GPU is available")
    
    # Repository parameters
    parser.add_argument("--repo_url", type=str, 
                       default="https://github.com/username/parkinsons-detection.git",
                       help="URL of the Git repository")
    parser.add_argument("--branch", type=str, default="main",
                       help="Branch of the repository to use")
    
    args = parser.parse_args()
    
    # Log starting configuration
    logger.info(f"Starting with configuration: {vars(args)}")
    
    # Setup Colab environment
    if not setup_colab_env():
        logger.error("Failed to setup Colab environment")
        return 1
    
    # Clone repository if needed (commented out for standalone scripts)
    # if not clone_repo(args.repo_url, args.branch):
    #     logger.error("Failed to clone repository")
    #     return 1
    
    # Run training pipeline
    if not run_training_pipeline(args):
        logger.error("Training pipeline failed")
        return 1
    
    logger.info("Parkinson's Disease Detection pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 