#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the Parkinson's Disease Detection model.

This script handles the training process for the model, including
data loading, training loop, validation, and logging.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import nibabel as nib
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.transformers.region_attention_transformer import (
    create_region_attention_transformer,
    RegionAttentionTransformer,
    RegionContrastiveTransformer,
    MultitaskLoss
)
from data.dataset import PDMRIDataset, MRIDataTransform, SimulatedMRIDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger('pd_training')

# Define paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
METADATA_DIR = ROOT_DIR / 'data' / 'metadata'
MODELS_DIR = ROOT_DIR / 'models' / 'pretrained'
RESULTS_DIR = ROOT_DIR / 'training' / 'results'

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for metric learning.
    
    This loss encourages embeddings of the same class to be close
    and embeddings of different classes to be far apart.
    """
    def __init__(self, margin=0.5, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size)
            
        Returns:
            loss: Scalar loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels matrix
        batch_size = labels.size(0)
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_matrix = label_matrix.float()
        
        # Mask out self-similarity
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        similarity_matrix = similarity_matrix.masked_fill(identity_mask, -float('inf'))
        
        # Compute positive and negative log-likelihood
        exp_sim = torch.exp(similarity_matrix)
        
        # For each row, compute the sum of exp(similarity) for the positive pairs
        pos_sim = torch.zeros_like(similarity_matrix)
        pos_sim = pos_sim.masked_fill(~identity_mask & label_matrix, 1.0)
        pos_exp_sim = exp_sim * pos_sim
        pos_sum = pos_exp_sim.sum(dim=1, keepdim=True)
        
        # Compute the denominator (sum of all exp(similarity))
        neg_sim = torch.ones_like(similarity_matrix)
        neg_sim = neg_sim.masked_fill(identity_mask, 0.0)
        den_sum = (exp_sim * neg_sim).sum(dim=1, keepdim=True)
        
        # Compute the loss
        epsilon = 1e-8  # Small constant to avoid numerical issues
        loss = -torch.log(pos_sum / (den_sum + epsilon) + epsilon)
        
        # Average over all samples
        loss = loss.mean()
        
        return loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_metrics, model_dir, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        val_metrics: Dictionary of validation metrics
        model_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_metrics': val_metrics
    }
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"pd_model_epoch{epoch}_{timestamp}.pt"
    filepath = os.path.join(model_dir, filename)
    
    # Save the checkpoint
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")
    
    # If this is the best model, create a symlink or copy
    if is_best:
        best_path = os.path.join(model_dir, 'best_model.pt')
        if os.path.exists(best_path):
            os.remove(best_path)
        torch.save(checkpoint, best_path)
        logger.info(f"Saved as best model to {best_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        epoch: Epoch of the checkpoint
        train_loss: Training loss at checkpoint
        val_loss: Validation loss at checkpoint
        val_metrics: Dictionary of validation metrics
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (
        checkpoint['epoch'],
        checkpoint['train_loss'],
        checkpoint['val_loss'],
        checkpoint['val_metrics']
    )


def evaluate_model(model, data_loader, device, criterion=None, use_contrastive=False):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        criterion: Loss function (optional)
        use_contrastive: Whether to use contrastive outputs
        
    Returns:
        metrics: Dictionary of evaluation metrics
        loss: Average loss (if criterion is provided)
        all_preds: List of all predictions
        all_labels: List of all ground truth labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            if use_contrastive:
                outputs, features, projections = model(inputs, return_features=True, return_projections=True)
            else:
                outputs = model(inputs)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                if use_contrastive:
                    loss, _ = criterion(outputs, labels, projections)
                else:
                    loss, _ = criterion(outputs, labels)
                total_loss += loss.item()
                batch_count += 1
            
            # Get predictions
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['precision'] = precision_score(all_labels, all_preds, average='weighted')
    metrics['recall'] = recall_score(all_labels, all_preds, average='weighted')
    metrics['f1'] = f1_score(all_labels, all_preds, average='weighted')
    
    # Calculate AUC if binary classification
    if model.num_classes == 2:
        metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
    
    # Calculate confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
    
    # Calculate average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    
    return metrics, avg_loss, all_preds, all_labels


def train_epoch(model, train_loader, optimizer, criterion, device, use_contrastive=False, use_regions=False):
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to run training on
        use_contrastive: Whether to use contrastive outputs
        use_regions: Whether to use region-specific outputs
        
    Returns:
        avg_loss: Average loss for the epoch
        loss_dict: Dictionary of average loss components
    """
    model.train()
    epoch_loss = 0.0
    epoch_loss_dict = {}
    batch_count = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if use_contrastive and use_regions:
            logits, _, projections, region_preds = model(
                inputs, 
                return_features=True,
                return_projections=True,
                return_region_preds=True
            )
            loss, loss_dict = criterion(logits, labels, projections, region_preds)
        elif use_contrastive:
            logits, _, projections = model(
                inputs, 
                return_features=True,
                return_projections=True
            )
            loss, loss_dict = criterion(logits, labels, projections)
        elif use_regions:
            logits, _, region_preds = model(
                inputs,
                return_features=True,
                return_region_preds=True
            )
            loss, loss_dict = criterion(logits, labels, None, region_preds)
        else:
            logits = model(inputs)
            loss, loss_dict = criterion(logits, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        epoch_loss += loss.item()
        batch_count += 1
        
        # Update component losses
        for k, v in loss_dict.items():
            if k in epoch_loss_dict:
                epoch_loss_dict[k] += v
            else:
                epoch_loss_dict[k] = v
    
    # Calculate averages
    avg_loss = epoch_loss / batch_count
    for k in epoch_loss_dict:
        epoch_loss_dict[k] /= batch_count
    
    return avg_loss, epoch_loss_dict


def plot_loss_curves(train_losses, val_losses, epoch_loss_components, save_path):
    """
    Plot and save loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        epoch_loss_components: Dictionary of loss components per epoch
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot main losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot loss components
    plt.subplot(1, 2, 2)
    for component, values in epoch_loss_components.items():
        if component != 'total':  # Skip total loss as it's in the main plot
            plt.plot(values, label=component)
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metrics(metrics_history, save_path):
    """
    Plot and save evaluation metrics.
    
    Args:
        metrics_history: Dictionary of metrics per epoch
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    # Extract metrics to plot
    plot_metrics = ['accuracy', 'precision', 'recall', 'f1']
    if 'auc' in metrics_history:
        plot_metrics.append('auc')
    
    for metric in plot_metrics:
        if metric in metrics_history:
            plt.plot(metrics_history[metric], label=metric.capitalize())
    
    plt.title('Evaluation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_dataset(data_dir, metadata_file, transform, is_train=True, use_simulated=False):
    """
    Create a dataset for training or validation.
    
    Args:
        data_dir: Directory containing data
        metadata_file: Path to metadata file
        transform: Transformations to apply
        is_train: Whether this is a training dataset
        use_simulated: Whether to use simulated data
        
    Returns:
        dataset: Dataset object
    """
    if use_simulated:
        # Use simulated data
        return SimulatedMRIDataset(
            input_size=(128, 128, 128),
            num_samples=1000 if is_train else 200,
            transform=transform,
            is_train=is_train
        )
    else:
        # Convert string paths to Path objects
        data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        metadata_file = Path(metadata_file) if isinstance(metadata_file, str) else metadata_file
        
        # Use real data
        return PDMRIDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            transform=transform,
            is_train=is_train
        )


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train Parkinson\'s Disease Detection Model')
    parser.add_argument('--data_dir', type=str, default=str(PROCESSED_DATA_DIR),
                        help='Directory containing processed MRI data')
    parser.add_argument('--metadata', type=str, default=str(METADATA_DIR / 'simulated_metadata.csv'),
                        help='Path to metadata file')
    parser.add_argument('--model_dir', type=str, default=str(MODELS_DIR),
                        help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', type=str, default=str(RESULTS_DIR),
                        help='Directory to save training results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension for transformer')
    parser.add_argument('--depth', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Use contrastive learning')
    parser.add_argument('--use_regions', action='store_true',
                        help='Use region-specific outputs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--simulated', action='store_true',
                        help='Use simulated data instead of real data')
    parser.add_argument('--input_size', type=int, default=128,
                        help='Input size of each dimension (assumed cube)')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch size for transformer')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create result directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(args.results_dir, f"run_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data transforms
    transform = MRIDataTransform(
        input_size=(args.input_size, args.input_size, args.input_size),
        is_train=True
    )
    
    val_transform = MRIDataTransform(
        input_size=(args.input_size, args.input_size, args.input_size),
        is_train=False
    )
    
    # Create datasets
    train_dataset = create_dataset(
        args.data_dir, 
        args.metadata, 
        transform, 
        is_train=True,
        use_simulated=args.simulated
    )
    
    val_dataset = create_dataset(
        args.data_dir, 
        args.metadata, 
        val_transform, 
        is_train=False,
        use_simulated=args.simulated
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders with {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    # Create model
    model = create_region_attention_transformer(
        input_size=(args.input_size, args.input_size, args.input_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=2,  # Binary classification (PD vs. Control)
        dropout=args.dropout,
        use_contrastive=args.use_contrastive
    )
    
    # Move model to device
    model = model.to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Define loss function
    criterion = MultitaskLoss(
        classification_weight=1.0,
        contrastive_weight=0.5 if args.use_contrastive else 0.0,
        region_weight=0.3 if args.use_regions else 0.0
    )
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize variables for training loop
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    metrics_history = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    epoch_loss_components = {}
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        start_epoch, prev_train_loss, prev_val_loss, prev_metrics = load_checkpoint(
            model, optimizer, args.resume
        )
        start_epoch += 1  # Start from the next epoch
        
        # Restore history
        train_losses = [prev_train_loss]
        val_losses = [prev_val_loss]
        for metric, value in prev_metrics.items():
            if metric in metrics_history and not isinstance(value, list):
                metrics_history[metric] = [value]
        
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss, epoch_loss_dict = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            use_contrastive=args.use_contrastive,
            use_regions=args.use_regions
        )
        
        # Evaluate on validation set
        val_metrics, val_loss, val_preds, val_labels = evaluate_model(
            model, val_loader, device, criterion, 
            use_contrastive=args.use_contrastive
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update loss components history
        for component, value in epoch_loss_dict.items():
            if component not in epoch_loss_components:
                epoch_loss_components[component] = []
            epoch_loss_components[component].append(value)
        
        # Update metrics history
        for metric, value in val_metrics.items():
            if metric in metrics_history and not isinstance(value, list):
                metrics_history[metric].append(value)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Metrics: Accuracy={val_metrics['accuracy']:.4f}, "
                   f"Precision={val_metrics['precision']:.4f}, "
                   f"Recall={val_metrics['recall']:.4f}, "
                   f"F1={val_metrics['f1']:.4f}")
        
        if 'auc' in val_metrics:
            logger.info(f"AUC: {val_metrics['auc']:.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint(
            model, optimizer, epoch + 1, train_loss, val_loss, val_metrics,
            args.model_dir, is_best=is_best
        )
        
        # Plot and save curves every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            # Save loss curves
            plot_loss_curves(
                train_losses, val_losses, epoch_loss_components,
                os.path.join(result_dir, f"loss_curves_epoch{epoch+1}.png")
            )
            
            # Save metrics curves
            plot_metrics(
                metrics_history,
                os.path.join(result_dir, f"metrics_epoch{epoch+1}.png")
            )
            
            # Save metrics as CSV
            metrics_df = pd.DataFrame(metrics_history)
            metrics_df.to_csv(os.path.join(result_dir, f"metrics_epoch{epoch+1}.csv"), index=True)
    
    logger.info("Training complete!")
    
    # Final evaluation on best model
    best_model_path = os.path.join(args.model_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        _, _, _, _ = load_checkpoint(model, optimizer, best_model_path)
        
        logger.info("Evaluating best model on validation set...")
        val_metrics, val_loss, val_preds, val_labels = evaluate_model(
            model, val_loader, device, criterion,
            use_contrastive=args.use_contrastive
        )
        
        # Log final metrics
        logger.info(f"Best model - Val Loss: {val_loss:.4f}")
        logger.info(f"Metrics: Accuracy={val_metrics['accuracy']:.4f}, "
                   f"Precision={val_metrics['precision']:.4f}, "
                   f"Recall={val_metrics['recall']:.4f}, "
                   f"F1={val_metrics['f1']:.4f}")
        
        if 'auc' in val_metrics:
            logger.info(f"AUC: {val_metrics['auc']:.4f}")
        
        # Save confusion matrix
        cm = np.array(val_metrics['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['Control', 'Parkinson\'s']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
        plt.close()
        
        # Save final metrics as JSON
        with open(os.path.join(result_dir, "final_metrics.json"), 'w') as f:
            json.dump(val_metrics, f, indent=4)


if __name__ == "__main__":
    main() 