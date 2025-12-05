import torch
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import csv
from datetime import datetime
from mil_net import Singletask_MILNET_status
from backbone_builder import BACKBONES
from dataset_loader import BreastDataset
import pandas as pd
from scipy import stats as st

def get_test_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for Status Classification")
    
    # Dataset args
    parser.add_argument("--test_json_path", default="./dataset/json/updated_test-type-0.json")
    parser.add_argument("--data_dir_path", required=True, help="Path to patches")
    parser.add_argument("--clinical_data_path", default="./dataset/clinical_data/preprocessed-type-0.xlsx")  
    parser.add_argument("--preloading", action="store_true")

    # Model args (Must match the trained model settings)
    parser.add_argument("--backbone", choices=BACKBONES, default="vgg16_bn")
    parser.add_argument("--dropout", type=float, default=0.2)
   
    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to best_status.pth")
    parser.add_argument("--num_workers", type=int, default=8)
    
    # for logging
    parser.add_argument("--csv_log_path", type=str, default="./evaluation_results.csv", 
                        help="Path to CSV file for logging results (will append if exists)")
    parser.add_argument("--model_name", type=str, default="", 
                        help="model name/identifier for logging")

    return parser.parse_args()

def calculate_multiclass_metrics(labels, probs, preds):
    """
    Calculates detailed metrics for multi-class classification (3 classes).
    Classes: N0 (0), N+(1-2) (1), N+(>2) (2)
    """
    metrics = {}
    
    # Basic Metrics
    metrics['accuracy'] = accuracy_score(labels, preds)
    
    # Macro-averaged metrics
    metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # Per-class F1 scores
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    metrics['f1_class_0'] = f1_per_class[0]  # N0
    metrics['f1_class_1'] = f1_per_class[1]  # N+(1-2)
    metrics['f1_class_2'] = f1_per_class[2]  # N+(>2)
    
    # AUC (one-vs-rest)
    try:
        metrics['auc_macro'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        metrics['auc_weighted'] = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except ValueError as e:
        print(f"Warning: Could not compute AUC: {e}")
        metrics['auc_macro'] = 0.0
        metrics['auc_weighted'] = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    metrics['confusion_matrix'] = cm
    
    # Per-class metrics from confusion matrix
    for i in range(3):
        # True Positives for class i
        tp = cm[i, i]
        # False Positives for class i (predicted as i but not actually i)
        fp = cm[:, i].sum() - tp
        # False Negatives for class i (actually i but not predicted as i)
        fn = cm[i, :].sum() - tp
        # True Negatives for class i
        tn = cm.sum() - tp - fp - fn
        
        # Sensitivity (Recall) for class i
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f'sensitivity_class_{i}'] = sensitivity
        
        # Specificity for class i
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f'specificity_class_{i}'] = specificity
        
        # Precision (PPV) for class i
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[f'precision_class_{i}'] = precision
        
        # NPV for class i
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics[f'npv_class_{i}'] = npv
    
    return metrics

def log_to_csv(args, status_acc, status_auc, note):
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
        'image_only': 'NA',
        'dropout': args.dropout,
        # Metastasis metrics
        'meta_accuracy': 'N/A',
        'meta_auc': 'N/A',
        'meta_f1': 'N/A',
        'meta_sensitivity': 'N/A',
        'meta_specificity': 'N/A',
        'meta_ppv': 'N/A',
        'meta_npv': 'N/A',
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
    For status: use mode (most common prediction) for patient-level prediction.
    """
    status_labels = []
    status_preds = []
    status_probs = []
    
    for p in agg_df['patient_id'].unique():
        patient_row = agg_df[agg_df['patient_id'] == p]
        # Use the maximum label (worst case) for ground truth
        status_labels.append(int(np.max(patient_row['status_label'])))
        # Use mode for status predictions
        status_mode = st.mode(patient_row['status_preds'], keepdims=False)
        status_preds.append(int(status_mode.mode))
        # Average probabilities across all bags for this patient
        status_probs.append(np.mean(patient_row['status_probs'].tolist(), axis=0))
    
    # Convert status_probs to numpy array
    status_probs = np.array(status_probs)
    
    return status_labels, status_probs, status_preds

def test(model, dataloader, args):
    model.eval()
    
    # Arrays to store results
    status_preds, status_labels, status_probs = [], [], []
    patient_ids = []
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for data in tqdm(dataloader, ncols=100):
            bag_tensor = data["bag_tensor"].cuda()
            clinical_data = data["clinical_data"].cuda()
            status_label = data["status_label"].cuda()
            
            # Forward pass
            status_logits, _ = model(bag_tensor, clinical_data)
            
            # Status (Multi-class: 3 classes)
            status_prob = torch.softmax(status_logits, dim=1)  # Probabilities for all classes
            status_pred = torch.argmax(status_logits, dim=1)
            
            status_probs.append(status_prob.cpu().numpy())
            status_preds.extend(status_pred.cpu().numpy())
            status_labels.extend(status_label.cpu().numpy())
            patient_ids.extend(data['patient_id'])

    # Convert to numpy
    status_probs = np.concatenate(status_probs, axis=0)
    status_preds = np.array(status_preds)
    status_labels = np.array(status_labels)
    patient_ids = np.array(patient_ids)
    
    def log_compute_metrics(status_labels, status_probs, status_preds, note="PER BAG LEVEL"):
        """
        Compute and log metrics for status task.
        """
        print(f"\n{note}")
        print("\n" + "="*40)
        print("  STATUS (Multi-class: 3 classes) RESULTS  ")
        print("="*40)
        print("Classes: 0=N0, 1=N+(1-2), 2=N+(>2)")
        print("="*40)
        
        status_metrics = calculate_multiclass_metrics(status_labels, status_probs, status_preds)
        
        # Print overall metrics
        print("\n--- Overall Metrics ---")
        print(f"{'ACCURACY':<20}: {status_metrics['accuracy']:.4f}")
        print(f"{'AUC_MACRO':<20}: {status_metrics['auc_macro']:.4f}")
        print(f"{'F1_MACRO':<20}: {status_metrics['f1_macro']:.4f}")
        
        # Print classification report
        print("\n--- Classification Report ---")
        target_names = ['N0', 'N+(1-2)', 'N+(>2)']
        print(classification_report(status_labels, status_preds, target_names=target_names, zero_division=0))
        
        return status_metrics['accuracy'], status_metrics['auc_macro']
    
    # Bag-level metrics
    status_acc_bag, status_auc_bag = log_compute_metrics(
        status_labels, status_probs, status_preds, 
        note="PER BAG LEVEL"
    )
    log_to_csv(args, status_acc_bag, status_auc_bag, note="PER BAG LEVEL")
    
    # Patient-level aggregation
    agg_dict = {
        'patient_id': patient_ids, 
        'status_preds': status_preds,
        'status_label': status_labels,
        'status_probs': list(status_probs)  # Store as list for patient-level aggregation
    }
    agg_df = pd.DataFrame(agg_dict)
    
    # Get patient-level metrics
    status_labels_patient, status_probs_patient, status_preds_patient = patient_level_metrics(agg_df)
    
    # Patient-level metrics
    status_acc_patient, status_auc_patient = log_compute_metrics(
        status_labels_patient, status_probs_patient, status_preds_patient, 
        note="PER PATIENT LEVEL"
    )
    log_to_csv(args, status_acc_patient, status_auc_patient, note="PER PATIENT LEVEL")

if __name__ == "__main__":
    args = get_test_args()
    
    # Load Dataset
    test_dataset = BreastDataset(args.test_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # Initialize Model
    print(f"Initializing model with backbone: {args.backbone}")
    model = Singletask_MILNET_status(backbone_name=args.backbone, dropout=args.dropout)
    model = model.cuda()
    
    # Load Checkpoint
    if os.path.isfile(args.checkpoint_path):
        print(f"Loading weights from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint_path}")
    
    # Run Test
    test(model, test_loader, args)
