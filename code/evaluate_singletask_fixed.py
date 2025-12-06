import torch
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import csv
from datetime import datetime
from mil_net import Singletask_MILNET
from backbone_builder import BACKBONES
from dataset_loader import BreastDataset
import pandas as pd

def get_test_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    
    # Dataset args
    parser.add_argument("--test_json_path", default="./dataset/json/updated_test-type-0.json")
    parser.add_argument("--data_dir_path", required=True, help="Path to patches")
    parser.add_argument("--clinical_data_path", default="./dataset/clinical_data/preprocessed-type-0.xlsx")  
    parser.add_argument("--preloading", action="store_true")

    # Model args (Must match the trained model settings)
    parser.add_argument("--backbone", choices=BACKBONES, default="vgg16_bn")
    parser.add_argument("--dropout", type=float, default=0.2)
   
    
    # Checkpoint
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to best_combined.pth or best_meta.pth")
    parser.add_argument("--num_workers", type=int, default=8)
    
    # for logging
    parser.add_argument("--csv_log_path", type=str, default="./evaluation_results.csv", 
                        help="Path to CSV file for logging results (will append if exists)")
    parser.add_argument("--model_name", type=str, default="", 
                        help="model name/identifier for logging")

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

def log_to_csv(args, meta_metrics, note):
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
        'meta_accuracy': meta_metrics['accuracy'],
        'meta_auc': meta_metrics['auc'],
        'meta_f1': meta_metrics['f1'],
        'meta_sensitivity': meta_metrics['sensitivity'],
        'meta_specificity': meta_metrics['specificity'],
        'meta_ppv': meta_metrics['ppv'],
        'meta_npv': meta_metrics['npv'],
        # Status metrics
        'status_accuracy': 'N/A',
        'status_auc_macro': 'N/A',
        'note': note
    }
    
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
    For metastasis: if any bag from a patient is positive, the patient is positive.
    """
    meta_labels = []
    meta_preds = []
    meta_probs = []
    
    for p in agg_df['patient_id'].unique():
        patient_row = agg_df[agg_df['patient_id'] == p]
        meta_probs.append(float(np.mean(patient_row['meta_probs'])))
        # If any instance of the patient is positive, then the patient is positive
        meta_labels.append(1 if float(np.mean(patient_row['meta_probs'])) > 0.5 else 0 ) #int(np.max(patient_row['meta_label']))
        meta_preds.append(int(np.max(patient_row['meta_preds'])))
        # Average probability across all bags for this patient
       # meta_probs.append(float(np.mean(patient_row['meta_probs'])))
    
    return meta_labels, meta_probs, meta_preds

def test(model, dataloader, args):
    model.eval()
    
    # Arrays to store results
    meta_preds, meta_labels, meta_probs = [], [], []
    patient_ids = []
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for data in tqdm(dataloader, ncols=100):
            bag_tensor = data["bag_tensor"].cuda()
            clinical_data = data["clinical_data"].cuda()
            metastasis_label = data["metastasis_label"].cuda()
            
            # Forward pass
            meta_logits, _ = model(bag_tensor, clinical_data)
            
            # Metastasis (Binary)
            meta_prob = torch.softmax(meta_logits, dim=1)[:, 1] # Probability of class 1
            meta_pred = torch.argmax(meta_logits, dim=1)
            
            meta_probs.extend(meta_prob.cpu().numpy())
            meta_preds.extend(meta_pred.cpu().numpy())
            meta_labels.extend(metastasis_label.cpu().numpy())
            patient_ids.extend(data['patient_id'])

    # Convert to numpy
    meta_probs = np.array(meta_probs)
    meta_preds = np.array(meta_preds)
    meta_labels = np.array(meta_labels)
    patient_ids = np.array(patient_ids)
    
    def log_compute_metrics(meta_labels, meta_probs, meta_preds, note="PER BAG LEVEL"):
        """
        Compute and log metrics for metastasis task.
        """
        print(f"\n{note}")
        print("\n" + "="*30)
        print("  METASTASIS (Binary) RESULTS  ")
        print("="*30)
        
        meta_metrics = calculate_extended_metrics(meta_labels, meta_probs, meta_preds)
        
        for k, v in meta_metrics.items():
            print(f"{k.upper():<15}: {v:.4f}")
        
        return meta_metrics
    
    # Bag-level metrics
    meta_metrics_bag = log_compute_metrics(meta_labels, meta_probs, meta_preds, note="PER BAG LEVEL")
    log_to_csv(args, meta_metrics_bag, note="PER BAG LEVEL")
    
    # Patient-level aggregation
    agg_dict = {
        'patient_id': patient_ids, 
        'meta_probs': meta_probs,
        'meta_preds': meta_preds,
        'meta_label': meta_labels
    }
    agg_df = pd.DataFrame(agg_dict)
    
    # Get patient-level metrics
    meta_labels_patient, meta_probs_patient, meta_preds_patient = patient_level_metrics(agg_df)
    
    # Patient-level metrics
    meta_metrics_patient = log_compute_metrics(
        meta_labels_patient, meta_probs_patient, meta_preds_patient, 
        note="PER PATIENT LEVEL"
    )
    log_to_csv(args, meta_metrics_patient, note="PER PATIENT LEVEL")

if __name__ == "__main__":
    args = get_test_args()
    
    test_dataset = BreastDataset(args.test_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
   
    print(f"Initializing model with backbone: {args.backbone}")
    model = Singletask_MILNET(backbone_name=args.backbone, dropout=args.dropout)
    model = model.cuda()
    
    if os.path.isfile(args.checkpoint_path):
        print(f"Loading weights from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint_path}")
    
    test(model, test_loader, args)
