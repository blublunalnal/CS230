"""
Quick script to compute optimal threshold using Youden's Index.
Minimal version without extensive plotting - for quick threshold determination.
Now saves results to JSON for use in evaluation script.
"""

import torch
import argparse
import numpy as np
import json
import os
from sklearn.metrics import roc_curve, accuracy_score, f1_score

from mil_net import Multitask_MILNET, Multitask_MILNET_image_only, Multitask_MILNET_shared_layer
from dataset_loader import BreastDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--val_json_path", default="./dataset/json/updated_val-type-0.json")
    parser.add_argument("--data_dir_path", required=True)
    parser.add_argument("--clinical_data_path", default="./dataset/clinical_data/preprocessed-type-0.xlsx")
    parser.add_argument("--backbone", default="vgg16_bn")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--image_only", action="store_true")
    parser.add_argument("--shared_layer", action="store_true")
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_json", type=str, default="./threshold_results.json",
                        help="Path to save threshold results JSON")
    return parser.parse_args()


def compute_optimal_threshold(labels, probs):
    """Compute optimal threshold using Youden's Index."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    
    optimal_threshold = thresholds[optimal_idx]
    optimal_preds = (probs >= optimal_threshold).astype(int)
    
    return {
        'threshold': float(optimal_threshold),
        'youden_index': float(youden_index[optimal_idx]),
        'sensitivity': float(tpr[optimal_idx]),
        'specificity': float(1 - fpr[optimal_idx]),
        'accuracy': float(accuracy_score(labels, optimal_preds)),
        'f1_score': float(f1_score(labels, optimal_preds, average='binary', zero_division=0))
    }


def main():
    args = parse_args()
    
    # Load model
    if args.image_only:
        model = Multitask_MILNET_image_only(backbone_name=args.backbone, dropout=args.dropout)
    elif args.shared_layer:
        model = Multitask_MILNET_shared_layer(backbone_name=args.backbone, dropout=args.dropout)
    else:
        model = Multitask_MILNET(backbone_name=args.backbone, dropout=args.dropout)
    
    checkpoint = torch.load(args.checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only= False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load data
    val_dataset = BreastDataset(args.val_json_path, args.data_dir_path, 
                                args.clinical_data_path, is_preloading=args.preloading)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False, 
        pin_memory=True, num_workers=args.num_workers
    )
    
    # Get predictions
    metastasis_probs = []
    metastasis_labels = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for data in val_loader:
            bag_tensor = data["bag_tensor"].to(device)
            clinical_data = data["clinical_data"].to(device)
            metastasis_label = data["metastasis_label"].cpu().numpy()
            
            if args.image_only:
                metastasis_logits, _, _ = model(bag_tensor)
            else:
                metastasis_logits, _, _ = model(bag_tensor, clinical_data)
            
            metastasis_prob = torch.softmax(metastasis_logits, dim=1)[:, 1].cpu().numpy()
            
            metastasis_probs.append(metastasis_prob)
            metastasis_labels.append(metastasis_label)
    
    metastasis_probs = np.concatenate(metastasis_probs)
    metastasis_labels = np.concatenate(metastasis_labels)
    
    print(f"\nValidation set size: {len(metastasis_labels)}")
    print(f"Class distribution: {np.bincount(metastasis_labels)}")
    
    # Compute optimal threshold
    result = compute_optimal_threshold(metastasis_labels, metastasis_probs)
    
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLD (Youden's Index)")
    print("="*60)
    print(f"Threshold:    {result['threshold']:.4f}")
    print(f"Youden Index: {result['youden_index']:.4f}")
    print(f"Sensitivity:  {result['sensitivity']:.4f}")
    print(f"Specificity:  {result['specificity']:.4f}")
    print(f"Accuracy:     {result['accuracy']:.4f}")
    print(f"F1 Score:     {result['f1_score']:.4f}")
    print("="*60)
    
    # Save to JSON file in the format expected by evaluate script
    output_data = {
        'metastasis': result,
        'checkpoint_path': args.checkpoint_path,
        'validation_set': args.val_json_path,
        'backbone': args.backbone,
        'image_only': args.image_only,
        'shared_layer': args.shared_layer
    }
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nThreshold results saved to: {args.output_json}")
    print(f"Use this file with evaluate_multitask_with_threshold.py:")
    print(f"  --threshold_json {args.output_json}")


if __name__ == "__main__":
    main()
