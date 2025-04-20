#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Parkinson's Disease Detection Model Trainer for Google Colab

This script trains a high-performance 3D CNN model to detect Parkinson's disease
from MRI data. It's optimized for Google Colab with GPU acceleration and is
designed to achieve >90% classification accuracy.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import nibabel as nib
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger('pd_trainer')

# Deep 3D CNN model for Parkinson's detection
class DeepVoxelCNN(nn.Module):
    def __init__(self, input_shape=(1, 128, 128, 128), num_classes=2, dropout_rate=0.5):
        super(DeepVoxelCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            
            # Second convolutional block
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            
            # Third convolutional block
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            
            # Fourth convolutional block
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            
            # Fifth convolutional block - focus on smaller features
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # Calculate the size after convolutions and pooling
        conv_output_size = self._get_conv_output(input_shape)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_output(self, shape):
        """Calculate the output size of the convolution layers"""
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
    def _initialize_weights(self):
        """Initialize model weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.features(x)
        x = self.classifier(x)
        return x

# Dataset class for MRI data
class MRIDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, augment=False, input_size=(128, 128, 128)):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.input_size = input_size
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load MRI data
        mri_path = self.file_paths[idx]
        img = nib.load(mri_path).get_fdata()
        
        # Ensure correct dimensions
        if img.shape != self.input_size:
            # Resize using nearest neighbor interpolation (simple approach)
            x_ratio = self.input_size[0] / img.shape[0]
            y_ratio = self.input_size[1] / img.shape[1]
            z_ratio = self.input_size[2] / img.shape[2]
            
            resized_img = np.zeros(self.input_size)
            for x in range(self.input_size[0]):
                for y in range(self.input_size[1]):
                    for z in range(self.input_size[2]):
                        resized_img[x, y, z] = img[int(x/x_ratio), int(y/y_ratio), int(z/z_ratio)]
            img = resized_img
        
        # Normalize to [0, 1]
        if img.max() > 0:
            img = img / img.max()
        
        # Apply data augmentation if enabled
        if self.augment:
            # Random flip
            if random.random() > 0.5:
                img = np.flip(img, axis=0)
            if random.random() > 0.5:
                img = np.flip(img, axis=1)
            
            # Random intensity adjustment
            if random.random() > 0.5:
                gamma = random.uniform(0.8, 1.2)
                img = np.power(img, gamma)
            
            # Random noise
            if random.random() > 0.5:
                noise_level = random.uniform(0.01, 0.03)
                noise = np.random.normal(0, noise_level, img.shape)
                img = img + noise
                img = np.clip(img, 0, 1)
            
            # Random shift
            if random.random() > 0.5:
                shift_x = random.randint(-4, 4)
                shift_y = random.randint(-4, 4)
                shift_z = random.randint(-4, 4)
                
                # Apply shift with zero padding
                if shift_x > 0:
                    img = np.pad(img[:-shift_x, :, :], ((shift_x, 0), (0, 0), (0, 0)), mode='constant')
                elif shift_x < 0:
                    img = np.pad(img[-shift_x:, :, :], ((0, -shift_x), (0, 0), (0, 0)), mode='constant')
                
                if shift_y > 0:
                    img = np.pad(img[:, :-shift_y, :], ((0, 0), (shift_y, 0), (0, 0)), mode='constant')
                elif shift_y < 0:
                    img = np.pad(img[:, -shift_y:, :], ((0, 0), (0, -shift_y), (0, 0)), mode='constant')
                
                if shift_z > 0:
                    img = np.pad(img[:, :, :-shift_z], ((0, 0), (0, 0), (shift_z, 0)), mode='constant')
                elif shift_z < 0:
                    img = np.pad(img[:, :, -shift_z:], ((0, 0), (0, 0), (0, -shift_z)), mode='constant')
        
        # Add channel dimension and convert to tensor
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, label

def create_datasets(data_dir, metadata_file, test_size=0.2, val_size=0.15):
    """Create train, validation, and test datasets"""
    logger.info(f"Creating datasets from {data_dir} using metadata {metadata_file}")
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Collect file paths and labels
    file_paths = []
    labels = []
    
    for subject in metadata['subjects']:
        subject_dir = os.path.join(data_dir, subject['subject_id'])
        mri_path = os.path.join(subject_dir, f"{subject['subject_id']}_T1.nii.gz")
        
        if os.path.exists(mri_path):
            file_paths.append(mri_path)
            labels.append(1 if subject['group'] == 'PD' else 0)
    
    logger.info(f"Found {len(file_paths)} subjects ({sum(labels)} PD, {len(labels) - sum(labels)} Control)")
    
    # Split into train, validation, and test sets
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        file_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=val_size/(1-test_size), 
        stratify=train_val_labels, random_state=42
    )
    
    logger.info(f"Split into {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
    
    # Create datasets - apply augmentation only to training data
    train_dataset = MRIDataset(train_files, train_labels, augment=True)
    val_dataset = MRIDataset(val_files, val_labels, augment=False)
    test_dataset = MRIDataset(test_files, test_labels, augment=False)
    
    return train_dataset, val_dataset, test_dataset

def train_model(model, train_loader, val_loader, test_loader, args):
    """Train the model with advanced techniques"""
    device = torch.device(args.device)
    model = model.to(device)
    logger.info(f"Training on {device}")
    
    # Setup training parameters
    if args.class_weights:
        weight = torch.tensor(args.class_weights, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        logger.info(f"Using weighted loss with weights {args.class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    # Mixed precision training
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    
    # Training variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    early_stop_count = 0
    
    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler is not None:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100 * train_correct / train_total if train_total > 0 else 0
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_true = []
        
        # Progress bar for validation
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Collect predictions for metrics
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': val_loss / (pbar.n + 1),
                    'acc': 100 * val_correct / val_total if val_total > 0 else 0
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Calculate validation metrics
        val_precision = precision_score(val_true, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_true, val_preds, average='weighted')
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        
        # Update learning rate scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                   f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        # Save checkpoint if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save model
            checkpoint_path = os.path.join(results_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None
            }, checkpoint_path)
            
            logger.info(f"Saved best model at epoch {epoch+1} with val_loss: {val_loss:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        # Save training curves periodically
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            # Plot loss curves
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Validation Accuracy')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"training_curves_epoch_{epoch+1}.png"))
            plt.close()
        
        # Early stopping
        if args.early_stopping > 0 and early_stop_count >= args.early_stopping:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(results_dir, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    logger.info("Evaluating best model on test set...")
    test_results = evaluate_model(model, test_loader, device, results_dir)
    
    # Save training configuration
    config = vars(args)
    config['best_epoch'] = best_epoch
    config['best_val_loss'] = float(best_val_loss)
    config['best_val_acc'] = float(best_val_acc)
    config['test_results'] = test_results
    
    with open(os.path.join(results_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    return model, test_results

def evaluate_model(model, test_loader, device, results_dir):
    """Evaluate the model on the test set"""
    model.eval()
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of PD
    
    test_loss /= len(test_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Try to calculate ROC AUC, which requires probabilities
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        logger.warning(f"Could not calculate ROC AUC: {e}")
        roc_auc = 0.0
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create results dictionary
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist()
    }
    
    # Log results
    logger.info(f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
               f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # Create visualizations
    plt.figure(figsize=(12, 5))
    
    # Plot confusion matrix
    plt.subplot(1, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Control', 'PD'])
    plt.yticks(tick_marks, ['Control', 'PD'])
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Plot ROC curve
    plt.subplot(1, 2, 2)
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
    except Exception as e:
        logger.warning(f"Could not plot ROC curve: {e}")
        plt.text(0.5, 0.5, 'ROC curve not available', 
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_results.png"))
    plt.close()
    
    # Also save the results as JSON
    with open(os.path.join(results_dir, "test_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def generate_synthetic_data(args):
    """Run the synthetic data generation script"""
    logger.info(f"Generating {args.num_subjects} synthetic subjects")
    
    # Run the synthetic data generation script
    cmd = [
        "python", 
        os.path.join(os.path.dirname(__file__), "pd_synthetic_data.py"),
        "--num_subjects", str(args.num_subjects),
        "--output_dir", args.data_dir,
        "--pd_ratio", str(args.pd_ratio),
        "--feature_strength", str(args.feature_strength),
        "--contrast_enhance", str(args.contrast_enhance)
    ]
    
    if args.visualize:
        cmd.append("--visualize")
        cmd.extend(["--vis_dir", args.vis_dir])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        logger.info(line.strip())
    
    return process.wait() == 0

def run_training_pipeline(args):
    """Run the complete training pipeline"""
    import os  # Add local import to ensure it's available
    start_time = time.time()
    logger.info("Starting Parkinson's Disease Detection Training Pipeline")
    logger.info(f"Configuration: {vars(args)}")
    
    # Step 1: Create necessary directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Step 2: Generate synthetic data
    if args.generate_data:
        logger.info("Step 1: Generating synthetic data")
        if not generate_synthetic_data(args):
            logger.error("Failed to generate synthetic data")
            return False
    else:
        logger.info("Skipping data generation step")
    
    # Step 3: Create datasets
    logger.info("Step 2: Creating datasets")
    metadata_file = os.path.join("data", "metadata", "simulated_metadata.json")
    train_dataset, val_dataset, test_dataset = create_datasets(
        args.data_dir, metadata_file, args.test_size, args.val_size
    )
    
    # Step 4: Create data loaders
    logger.info("Step 3: Creating data loaders")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Step 5: Create model
    logger.info("Step 4: Creating model")
    model = DeepVoxelCNN(dropout_rate=args.dropout)
    
    # Step 6: Train model
    logger.info("Step 5: Training model")
    model, test_results = train_model(
        model, train_loader, val_loader, test_loader, args
    )
    
    # Log total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Step 7: Save results to Google Drive if in Colab
    if args.save_to_drive:
        logger.info("Step 6: Saving results to Google Drive")
        try:
            from google.colab import drive
            drive_mounted = True
        except ImportError:
            drive_mounted = False
            logger.warning("Not running in Google Colab, skipping Drive save")
        
        if drive_mounted:
            # Mount Drive if not already mounted
            import os
            if not os.path.exists("/content/drive"):
                drive.mount('/content/drive')
            
            # Create directory for results
            drive_path = f"/content/drive/MyDrive/parkinson_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(drive_path, exist_ok=True)
            
            # Copy results
            timestamp_dirs = [d for d in os.listdir(args.save_dir) if d.startswith("run_")]
            latest_dir = max(timestamp_dirs, key=lambda x: os.path.getmtime(os.path.join(args.save_dir, x)))
            latest_path = os.path.join(args.save_dir, latest_dir)
            
            subprocess.run(["cp", "-r", latest_path, drive_path])
            subprocess.run(["cp", "-r", os.path.join(args.vis_dir), drive_path])
            subprocess.run(["cp", metadata_file, drive_path])
            
            logger.info(f"Results saved to Google Drive: {drive_path}")
    
    # Return success
    return True

def main():
    """Parse arguments and run training pipeline"""
    import os  # Add local import to ensure it's available
    parser = argparse.ArgumentParser(description="Train Parkinson's disease detection model")
    
    # Data generation parameters
    parser.add_argument("--generate_data", action="store_true", help="Generate synthetic data")
    parser.add_argument("--num_subjects", type=int, default=1000, help="Number of subjects to generate")
    parser.add_argument("--pd_ratio", type=float, default=0.5, help="Ratio of PD cases to generate")
    parser.add_argument("--feature_strength", type=float, default=8.0, 
                        help="Strength of disease features (0-10)")
    parser.add_argument("--contrast_enhance", type=float, default=6.0, 
                        help="Contrast enhancement factor")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of slices")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/raw/improved", 
                        help="Directory containing MRI data")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Proportion of data to use for testing")
    parser.add_argument("--val_size", type=float, default=0.15, 
                        help="Proportion of remaining data to use for validation")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=1e-7, 
                        help="Minimum learning rate for scheduler")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--class_weights", type=float, nargs="+", default=[2.5, 1.0], 
                        help="Class weights for loss function")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"], 
                        help="Learning rate scheduler")
    parser.add_argument("--early_stopping", type=int, default=20, 
                        help="Stop training if validation loss doesn't improve for this many epochs")
    parser.add_argument("--mixed_precision", action="store_true", 
                        help="Use mixed precision training")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to train on")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    parser.add_argument("--save_dir", type=str, default="training/results", 
                        help="Directory to save results")
    parser.add_argument("--vis_dir", type=str, default="visualizations/synthetic", 
                        help="Directory to save visualizations")
    parser.add_argument("--save_to_drive", action="store_true", 
                        help="Save results to Google Drive (Colab)")
    
    args = parser.parse_args()
    
    # Run the training pipeline
    success = run_training_pipeline(args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 