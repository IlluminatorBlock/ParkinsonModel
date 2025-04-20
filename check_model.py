#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to evaluate a trained Parkinson's disease detection model
and generate detailed performance metrics and visualizations.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import PDMRIDataset, MRIDataTransform

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Parkinson's disease detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="visualizations/evaluation", 
                        help="Directory to save evaluation results")
    parser.add_argument("--detailed_metrics", action="store_true", 
                        help="Generate detailed performance metrics")
    parser.add_argument("--confusion_matrix", action="store_true", 
                        help="Generate confusion matrix visualization")
    parser.add_argument("--roc_curve", action="store_true", 
                        help="Generate ROC curve")
    parser.add_argument("--class_names", type=str, nargs='+', default=["Control", "Parkinson's"],
                        help="Class names for visualization")
    return parser.parse_args()

def create_dataset(data_dir):
    """Create dataset from directory of preprocessed MRI data"""
    # Check if metadata file exists
    metadata_path = os.path.join(os.path.dirname(data_dir), "test_metadata.csv")
    
    if os.path.exists(metadata_path):
        # Use metadata file
        df = pd.read_csv(metadata_path)
        file_paths = [os.path.join(data_dir, fp) for fp in df["file_path"]]
        labels = [1 if group == "PD" else 0 for group in df["group"]]
        subject_ids = df["subject_id"].tolist()
    else:
        # Scan directory for files
        file_paths = []
        labels = []
        subject_ids = []
        
        pd_dir = os.path.join(data_dir, "PD")
        control_dir = os.path.join(data_dir, "Control")
        
        # Add PD subjects
        if os.path.exists(pd_dir):
            for filename in os.listdir(pd_dir):
                if filename.endswith(".nii.gz") or filename.endswith(".nii"):
                    file_paths.append(os.path.join(pd_dir, filename))
                    labels.append(1)  # PD class
                    subject_ids.append(os.path.splitext(filename)[0])
        
        # Add Control subjects
        if os.path.exists(control_dir):
            for filename in os.listdir(control_dir):
                if filename.endswith(".nii.gz") or filename.endswith(".nii"):
                    file_paths.append(os.path.join(control_dir, filename))
                    labels.append(0)  # Control class
                    subject_ids.append(os.path.splitext(filename)[0])
        
        # If no subdirectories, scan directly
        if not file_paths:
            for filename in os.listdir(data_dir):
                if filename.endswith(".nii.gz") or filename.endswith(".nii"):
                    file_paths.append(os.path.join(data_dir, filename))
                    # Try to infer label from filename
                    if "pd" in filename.lower() or "parkinson" in filename.lower():
                        labels.append(1)  # PD class
                    else:
                        labels.append(0)  # Assume Control
                    subject_ids.append(os.path.splitext(filename)[0])
    
    # Create transform
    transform = MRIDataTransform(input_size=(128, 128, 128), is_train=False)
    
    # Create dataset
    dataset = PDMRIDataset(file_paths, labels, transform=transform, subject_ids=subject_ids)
    
    return dataset, subject_ids

def load_model(model_path):
    """Load the trained model from the specified path"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_data = torch.load(model_path, map_location=device)
    
    if isinstance(model_data, dict) and "model_state_dict" in model_data:
        # This is a checkpoint dictionary
        # We need to determine model architecture from the state_dict
        state_dict = model_data["model_state_dict"]
        
        # Check for transformer architecture (region attention)
        if any("transformer" in key for key in state_dict.keys()):
            from models.transformers.region_attention_transformer import create_region_attention_transformer
            model = create_region_attention_transformer(
                input_size=(128, 128, 128),
                patch_size=16,
                num_classes=2
            )
        else:
            # Assume deep_voxel_3d architecture
            model = torch.nn.Sequential(
                torch.nn.Conv3d(1, 32, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(32),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2),
                
                torch.nn.Conv3d(32, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2),
                
                torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2),
                
                torch.nn.Conv3d(128, 256, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(256),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2),
                
                torch.nn.Conv3d(256, 512, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(512),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2),
                
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, 2)
            )
        
        # Load state dict
        model.load_state_dict(state_dict)
    else:
        # This is the model itself
        model = model_data
    
    model = model.to(device)
    model.eval()
    return model, device

def evaluate_model(model, dataset, device, batch_size=8):
    """Evaluate the model on the test dataset and return predictions and probabilities"""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_subject_ids = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions and probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Convert to numpy arrays
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            # Append to lists
            all_preds.extend(preds_np)
            all_probs.extend(probs_np)
            all_labels.extend(labels.numpy())
            
            # Get subject IDs for this batch
            batch_indices = list(range(i * batch_size, min((i + 1) * batch_size, len(dataset))))
            batch_subject_ids = [dataset.subject_ids[idx] if hasattr(dataset, 'subject_ids') else str(idx) for idx in batch_indices]
            all_subject_ids.extend(batch_subject_ids)
    
    return all_preds, all_probs, all_labels, all_subject_ids

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate performance metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # ROC AUC (only if we have both classes)
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        roc_auc = float('nan')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Precision and recall for each class
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": class_report
    }

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, output_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, output_path):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def plot_prediction_distributions(y_true, y_prob, class_names, output_path):
    """Plot distributions of prediction probabilities for each class"""
    plt.figure(figsize=(12, 6))
    
    # Split by true class
    for i, class_name in enumerate(class_names):
        class_probs = y_prob[np.array(y_true) == i, 1]
        if len(class_probs) > 0:
            sns.kdeplot(class_probs, label=f"True {class_name}", fill=True, alpha=0.5)
    
    plt.xlabel("Probability of Parkinson's Disease")
    plt.ylabel("Density")
    plt.title("Distribution of Prediction Probabilities by Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def analyze_errors(y_true, y_pred, y_prob, subject_ids, output_path):
    """Analyze and save information about misclassified samples"""
    # Create DataFrame with all predictions
    df = pd.DataFrame({
        "subject_id": subject_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "prob_control": y_prob[:, 0],
        "prob_pd": y_prob[:, 1],
        "correct": np.array(y_true) == np.array(y_pred)
    })
    
    # Filter for incorrect predictions
    errors_df = df[~df["correct"]].copy()
    
    # Sort by prediction confidence (highest first)
    errors_df["confidence"] = np.maximum(errors_df["prob_control"], errors_df["prob_pd"])
    errors_df = errors_df.sort_values("confidence", ascending=False)
    
    # Save to CSV
    errors_df.to_csv(output_path, index=False)
    
    return errors_df

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test dataset
    print(f"Loading test data from {args.test_data}...")
    test_dataset, subject_ids = create_dataset(args.test_data)
    print(f"Created dataset with {len(test_dataset)} samples")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, device = load_model(args.model_path)
    
    # Evaluate model
    print("Evaluating model...")
    predictions, probabilities, true_labels, subject_ids = evaluate_model(
        model, test_dataset, device, batch_size=args.batch_size
    )
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(true_labels, predictions, probabilities)
    
    # Print and save summary metrics
    print("\n=== Model Performance ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Convert metrics to serializable format
    serializable_metrics = {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "roc_auc": float(metrics["roc_auc"]),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "n_samples": len(true_labels),
        "model_path": args.model_path,
        "test_data": args.test_data
    }
    
    # Save metrics as JSON
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(serializable_metrics, f, indent=4)
    
    # Generate visualizations if requested
    if args.confusion_matrix:
        print("Generating confusion matrix...")
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(metrics["confusion_matrix"], args.class_names, cm_path)
    
    if args.roc_curve:
        print("Generating ROC curve...")
        roc_path = os.path.join(args.output_dir, "roc_curve.png")
        plot_roc_curve(true_labels, probabilities, roc_path)
        
        # Also generate precision-recall curve
        pr_path = os.path.join(args.output_dir, "precision_recall_curve.png")
        plot_precision_recall_curve(true_labels, probabilities, pr_path)
    
    if args.detailed_metrics:
        print("Generating detailed performance analysis...")
        
        # Save full classification report
        with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
            f.write(classification_report(true_labels, predictions, target_names=args.class_names))
        
        # Plot prediction distributions
        dist_path = os.path.join(args.output_dir, "prediction_distributions.png")
        plot_prediction_distributions(true_labels, probabilities, args.class_names, dist_path)
        
        # Analyze errors
        errors_path = os.path.join(args.output_dir, "misclassified_samples.csv")
        error_df = analyze_errors(true_labels, predictions, probabilities, subject_ids, errors_path)
        print(f"Found {len(error_df)} misclassified samples")
        
        # Save all predictions
        all_preds_df = pd.DataFrame({
            "subject_id": subject_ids,
            "true_label": true_labels,
            "predicted_label": predictions,
            "prob_control": probabilities[:, 0],
            "prob_pd": probabilities[:, 1]
        })
        all_preds_df.to_csv(os.path.join(args.output_dir, "all_predictions.csv"), index=False)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")
    
    # Return 1 if model performance meets threshold, 0 otherwise
    if metrics["f1_score"] >= 0.9 and metrics["accuracy"] >= 0.9:
        print("\n✅ Model performance meets desired thresholds (F1 score and accuracy ≥ 0.9)!")
        return 0
    else:
        print("\n⚠️ Model performance does not meet desired thresholds (F1 score and accuracy ≥ 0.9).")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 