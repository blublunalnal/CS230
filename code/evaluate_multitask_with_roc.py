import torch
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve
from tqdm import tqdm
import csv
from datetime import datetime
from mil_net import Multitask_MILNET, Multitask_MILNET_image_only, Multitask_MILNET_shared_layer
from backbone_builder import BACKBONES
from dataset_loader import BreastDataset
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import json

def get_test_args():
    parser = argparse.ArgumentParser(description="Multitask Evaluation Script with ROC Curves")
    
    # Dataset args
    parser.add_argument("--test_json_path", default="./dataset/json/updated_test-type-0.json")
    parser.add_argument("--data_dir_path", required=True, help="Path to patches")
    parser.add_argument("--clinical_data_path", default="./dataset/clinical_data/preprocessed-type-0.xlsx")  
    parser.add_argument("--preloading", action="store_true")

    # Model args (Must match the trained model settings)
    parser.add_argument("--backbone", choices=BACKBONES, default="vgg16_bn")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--image_only", action="store_true")
    
    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to best_combined.pth or best_meta.pth")
    parser.add_argument("--num_workers", type=int, default=8)
    
    # log to csv
    parser.add_argument("--csv_log_path", type=str, default="./evaluation_results.csv", 
                        help="Path to CSV file for logging results (will append if exists)")
    parser.add_argument("--model_name", type=str, default="", 
                        help="model name/identifier for logging")
    # load the model with shared layer
    parser.add_argument("--shared_layer", action="store_true", help="load multi-task model with a shared layer")
    
    # ROC curve options
    parser.add_argument("--output_dir", type=str, default="./evaluation_outputs",
                        help="Directory to save ROC curve plots")
    parser.add_argument("--save_roc_data", action="store_true",
                        help="Save ROC curve data (FPR, TPR, thresholds) as JSON")

    return parser.parse_args()

def calculate_extended_metrics(labels, probs, preds):
    """
    Calculates detailed metrics for binary classification.
    Assumes labels are 0 and 1.
    """
    metrics = {}
    
    # Basic Metrics
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['auc'] = roc_auc_score(labels, probs)
    metrics['f1'] = f1_score(labels, preds)
    
    # Confusion Matrix (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    # Extended Clinical Metrics
    # Sensitivity (Recall) = TP / (TP + FN)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['recall'] = metrics['sensitivity'] # Recall is same as Sensitivity
    
    # Specificity = TN / (TN + FP)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Positive Predictive Value (Precision) = TP / (TP + FP)
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['precision'] = metrics['ppv'] # Precision is same as PPV
    
    # Negative Predictive Value = TN / (TN + FN)
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return metrics

def plot_confusion_matrix(labels, preds, output_path, title="Confusion Matrix", 
                         class_names=None, normalize=False):
    """
    Plot and save confusion matrix.
    
    Args:
        labels: Ground truth labels
        preds: Predicted labels
        output_path: Path to save the plot
        title: Plot title
        class_names: List of class names for labels
        normalize: Whether to normalize the confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Auto-generate class names if not provided
    if class_names is None:
        unique_labels = sorted(np.unique(np.concatenate([labels, preds])))
        class_names = [f'Class {i}' for i in unique_labels]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = title + " (Normalized)"
    else:
        fmt = 'd'
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")

def plot_roc_curve(labels, probs, output_path, title="ROC Curve", auc_score=None):
    """
    Plot and save ROC curve.
    
    Args:
        labels: Ground truth labels
        probs: Predicted probabilities
        output_path: Path to save the plot
        title: Plot title
        auc_score: Pre-calculated AUC score (if None, will calculate)
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    if auc_score is None:
        auc_score = roc_auc_score(labels, probs)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {output_path}")
    
    return fpr, tpr, thresholds

def save_roc_data(labels, probs, output_path, level="bag", task="metastasis"):
    """
    Save ROC curve data as JSON for later analysis.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    
    roc_data = {
        'task': task,
        'level': level,
        'auc': float(auc_score),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'n_samples': len(labels),
        'n_positive': int(np.sum(labels)),
        'n_negative': int(len(labels) - np.sum(labels))
    }
    
    with open(output_path, 'w') as f:
        json.dump(roc_data, f, indent=2)
    
    print(f"ROC data saved to: {output_path}")

def log_to_csv(args, meta_metrics, status_acc, status_auc, note):
    """
    Log evaluation results to CSV file (append mode).
    Creates the file with headers if it doesn't exist.
    """
    csv_path = args.csv_log_path
    file_exists = os.path.isfile(csv_path)
    
    # Prepare row data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_identifier = args.model_name if args.model_name else os.path.basename(args.checkpoint_path)
    
    row_data = {
        'timestamp': timestamp,
        'model_name': model_identifier,
        'checkpoint_path': args.checkpoint_path,
        'backbone': args.backbone,
        'image_only': args.image_only,
        'dropout': args.dropout,
        # Metastasis metrics
        'meta_accuracy': meta_metrics['accuracy'],
        'meta_auc': meta_metrics['auc'],
        'meta_f1': meta_metrics['f1'],
        'meta_sensitivity': meta_metrics['sensitivity'],
        'meta_specificity': meta_metrics['specificity'],
        'meta_ppv': meta_metrics['ppv'],
        'meta_npv': meta_metrics['npv'],
        # Status metrics
        'status_accuracy': status_acc,
        'status_auc_macro': status_auc,
        'note': note
    }
    
    # Define column order
    fieldnames = [
        'timestamp', 'model_name', 'checkpoint_path', 'backbone', 'image_only', 'dropout', 
        'meta_accuracy', 'meta_auc', 'meta_f1', 'meta_sensitivity', 'meta_specificity', 
        'meta_ppv', 'meta_npv', 'status_accuracy', 'status_auc_macro', 'note'
    ]
    
    # Write to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)
    
    print(f"\nResults logged to: {csv_path}")
    
def patient_level_metrics(agg_df):
    """
    Aggregate predictions and labels at patient level.
    """
    status_labels = []
    meta_labels = []
    meta_preds = []
    status_preds = []
    meta_probs = []
    status_probs = []
    
    for p in agg_df['patient_id'].unique():
        patient_row = agg_df[agg_df['patient_id'] == p]
        meta_labels.append(int(np.max(patient_row['meta_label'])))
        status_labels.append(int(np.max(patient_row['status_label'])))
       
        meta_probs.append(float(np.mean(patient_row['meta_probs'])))
        meta_preds.append( 1 if float(np.mean(patient_row['meta_probs'])) > 0.5 else 0 )
        # Use mode for status predictions
        status_mode = st.mode(patient_row['status_preds'], keepdims=False)
        status_preds.append(int(status_mode.mode))
        
        # For status probs, need to handle multi-class properly
        status_probs.append(np.mean(patient_row['status_probs'].tolist(), axis=0))
    
    # Convert status_probs to numpy array
    status_probs = np.array(status_probs)
    
    return meta_labels, meta_probs, meta_preds, status_labels, status_probs, status_preds

def test(model, dataloader, args):
    model.eval()
    
    # Create output directory for plots
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Arrays to store results
    meta_preds, meta_labels, meta_probs = [], [], []
    status_preds, status_labels, status_probs = [], [], []
    patient_ids = []
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for data in tqdm(dataloader, ncols=100):
            bag_tensor = data["bag_tensor"].cuda()
            clinical_data = data["clinical_data"].cuda()
            metastasis_label = data["metastasis_label"].cuda()
            status_label = data["status_label"].cuda()
            
            # Forward pass
            if args.image_only:
                meta_logits, status_logits, _ = model(bag_tensor)
            else:
                meta_logits, status_logits, _ = model(bag_tensor, clinical_data)
            
            # Metastasis (Binary) 
            meta_prob = torch.softmax(meta_logits, dim=1)[:, 1] # Probability of class 1
            meta_pred = torch.argmax(meta_logits, dim=1)

            meta_probs.extend(meta_prob.cpu().numpy())
            meta_preds.extend(meta_pred.cpu().numpy())
            meta_labels.extend(metastasis_label.cpu().numpy())
            
            # Status (Multiclass) 
            status_prob = torch.softmax(status_logits, dim=1)
            status_pred = torch.argmax(status_logits, dim=1)
            status_probs.append(status_prob.cpu().numpy())
            status_preds.extend(status_pred.cpu().numpy())
            status_labels.extend(status_label.cpu().numpy())
            patient_ids.extend(data['patient_id'])

    # Convert to numpy
    meta_probs = np.array(meta_probs)
    meta_preds = np.array(meta_preds)
    meta_labels = np.array(meta_labels)
    patient_ids = np.array(patient_ids)
    
    status_probs = np.concatenate(status_probs, axis=0)
    status_preds = np.array(status_preds)
    status_labels = np.array(status_labels)
    
    def log_compute_metrics(meta_labels, meta_probs, meta_preds, status_labels, status_probs, status_preds, note="per bag level", level="bag"):
        """
        Compute and log metrics for both tasks.
        """
        print(f"\n{note}")
        print("\n" + "="*30)
        print("  METASTASIS (Binary) RESULTS  ")
        print("="*30)
        
        meta_metrics = calculate_extended_metrics(meta_labels, meta_probs, meta_preds)
        
        for k, v in meta_metrics.items():
            print(f"{k.upper():<15}: {v:.4f}")

        # Generate model identifier for filenames
        model_id = args.model_name if args.model_name else os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        
        # Plot ROC curve for metastasis task
        roc_plot_path = os.path.join(args.output_dir, f"{model_id}_meta_roc_{level}_level.png")
        plot_roc_curve(
            meta_labels, 
            meta_probs, 
            roc_plot_path,
            title=f"ROC Curve - Metastasis Prediction ({level.upper()} Level)",
            auc_score=meta_metrics['auc']
        )
        
        # Plot confusion matrix for metastasis (unnormalized)
        cm_meta_plot_path = os.path.join(args.output_dir, f"{model_id}_meta_cm_{level}_level.png")
        plot_confusion_matrix(
            meta_labels,
            meta_preds,
            cm_meta_plot_path,
            title=f"Confusion Matrix - Metastasis Prediction ({level.upper()} Level)",
            class_names=['Negative', 'Positive'],
            normalize=False
        )
        
        # Plot confusion matrix for metastasis (normalized)
        cm_meta_norm_plot_path = os.path.join(args.output_dir, f"{model_id}_meta_cm_normalized_{level}_level.png")
        plot_confusion_matrix(
            meta_labels,
            meta_preds,
            cm_meta_norm_plot_path,
            title=f"Confusion Matrix - Metastasis Prediction ({level.upper()} Level)",
            class_names=['Negative', 'Positive'],
            normalize=True
        )
        
        # Optionally save ROC data for metastasis
        if args.save_roc_data:
            roc_data_path = os.path.join(args.output_dir, f"{model_id}_meta_roc_{level}_level.json")
            save_roc_data(meta_labels, meta_probs, roc_data_path, level=level, task="metastasis")

        print("\n" + "="*30)
        print("  STATUS (Multiclass) RESULTS  ")
        print("="*30)
        
        status_acc = accuracy_score(status_labels, status_preds)
        try:
            status_auc = roc_auc_score(status_labels, status_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Warning: Could not compute AUC for status task: {e}")
            status_auc = 0.0
            
        print(f"Accuracy       : {status_acc:.4f}")
        print(f"AUC (Macro)    : {status_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(status_labels, status_preds, digits=4))
        
        # Plot confusion matrix for status (unnormalized)
        # Determine unique class labels for proper naming
        unique_status = sorted(np.unique(np.concatenate([status_labels, status_preds])))
        status_class_names = [f'Status {i}' for i in unique_status]
        
        cm_status_plot_path = os.path.join(args.output_dir, f"{model_id}_status_cm_{level}_level.png")
        plot_confusion_matrix(
            status_labels,
            status_preds,
            cm_status_plot_path,
            title=f"Confusion Matrix - Status Prediction ({level.upper()} Level)",
            class_names=status_class_names,
            normalize=False
        )
        
        # Plot confusion matrix for status (normalized)
        cm_status_norm_plot_path = os.path.join(args.output_dir, f"{model_id}_status_cm_normalized_{level}_level.png")
        plot_confusion_matrix(
            status_labels,
            status_preds,
            cm_status_norm_plot_path,
            title=f"Confusion Matrix - Status Prediction ({level.upper()} Level)",
            class_names=status_class_names,
            normalize=True
        )
        
        return meta_metrics, status_acc, status_auc
        
    # Bag-level metrics
    meta_metrics_bag, status_acc_bag, status_auc_bag = log_compute_metrics(
        meta_labels, meta_probs, meta_preds, 
        status_labels, status_probs, status_preds, 
        note="PER BAG LEVEL",
        level="bag"
    )
    
    # Log bag-level results to CSV
    log_to_csv(args, meta_metrics_bag, status_acc_bag, status_auc_bag, note = "PER BAG LEVEL")
    
    # Patient-level aggregation
    agg_dict = {
        'patient_id': patient_ids, 
        'meta_probs': meta_probs,
        'meta_preds': meta_preds,
        'status_preds': status_preds,
        'meta_label': meta_labels,
        'status_label': status_labels,
        'status_probs': list(status_probs)  # Store as list for patient-level aggregation
    }
    agg_df = pd.DataFrame(agg_dict)
    
    # Get patient-level metrics
    meta_labels_patient, meta_probs_patient, meta_preds_patient, \
    status_labels_patient, status_probs_patient, status_preds_patient = patient_level_metrics(agg_df)
    
    # Patient-level metrics
    meta_metrics_patient, status_acc_patient, status_auc_patient = log_compute_metrics(
        meta_labels_patient, meta_probs_patient, meta_preds_patient, 
        status_labels_patient, status_probs_patient, status_preds_patient, 
        note="PER PATIENT LEVEL",
        level="patient"
    )
    
    # log to csv patient-level matrics
    log_to_csv(args, meta_metrics_patient, status_acc_patient, status_auc_patient, note = "PER PATIENT LEVEL")
    
   

if __name__ == "__main__":
    args = get_test_args()
    
    test_dataset = BreastDataset(
        args.test_json_path, 
        args.data_dir_path, 
        args.clinical_data_path, 
        is_preloading=args.preloading
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Initialize Model
    print(f"Initializing model with backbone: {args.backbone}")
    if args.image_only:
        model = Multitask_MILNET_image_only(backbone_name=args.backbone, dropout=args.dropout)
    elif args.shared_layer:
        model = Multitask_MILNET_shared_layer(backbone_name=args.backbone, dropout=args.dropout)
    else:
        model = Multitask_MILNET(backbone_name=args.backbone, dropout=args.dropout)
    
    model = model.cuda()
    
    # Load Checkpoint
    if os.path.isfile(args.checkpoint_path):
        print(f"Loading weights from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint_path}")
    
    test(model, test_loader, args)
