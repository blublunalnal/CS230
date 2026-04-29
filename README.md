# Multi-Task Learning with Attention-Based MIL for Breast Cancer Metastasis Prediction

**Stanford CS230 (Deep Learning) — Course Project**  
Author: Lan Lan &nbsp;|&nbsp; [GitHub](https://github.com/blublunalnal/CS230)

---

## Overview

Axillary lymph node (ALN) metastasis is one of the most critical prognostic indicators in early-stage breast cancer and directly determines the extent of surgery and adjuvant therapy. Confirming ALN status currently requires sentinel lymph node biopsy — an invasive surgical procedure. This project investigates whether a deep learning model trained on routine H&E-stained whole-slide images (WSI) and clinical data can non-invasively predict ALN status.

The core research question: **does joint multi-task learning of metastasis status and involvement stage outperform single-task approaches, and does it learn more robust representations?**

---

## Key Contributions

1. **Multi-task ABMIL framework** — Jointly predicts ALN metastasis (binary) and involvement stage (3-class: N0 / N+ 1–2 / N+ >2) from WSI bags, with a shared backbone and attention module.
2. **Pathology foundation model** — Integrates **UNI** (Chen et al., *Nature Medicine* 2024), a ViT-based model pre-trained on millions of pathology images, as a drop-in replacement for VGG16-BN, yielding consistent gains across all metrics.
3. **Multimodal clinical fusion** — Concatenates attention-aggregated image features with clinical variables (age, tumor size, tumor type, ER/PR/HER2 status) to exploit complementary signal.
4. **Systematic ablation** — Compares single-task vs. multi-task learning, UNI vs. VGG16-BN, and image-only vs. image+clinical fusion across five model configurations.

---

## Dataset — BCNB

| Property | Detail |
|---|---|
| Patients | 1,058 |
| Patch size | 256 × 256 pixels (non-overlapping ROI segments) |
| Bag structure | Multiple bags per patient (each bag = N randomly selected patches) |
| Clinical features | Age, tumor size, tumor type, ER / PR / HER2 status |
| Class distribution | 655 ALN-negative / 403 ALN-positive |
| Splits | Train 60% / Val 20% / Test 20% (stratified by patient outcome) |

---

## Architecture

```
WSI Patches (bag of N patches)
         │
         ▼
 ┌─────────────────────┐
 │   Feature Extractor │  UNI (frozen ViT, 1,536-dim)
 │     (per patch)     │  VGG16-BN (unfrozen, 25,088-dim)
 └─────────────────────┘
         │  [N × D] patch features
         ▼
 ┌──────────────────────────────────────┐
 │     Attention-Based MIL Pooling      │
 │  H = ReLU(X W₁ᵀ)  ∈ ℝᴺˣᴸ           │
 │  e = Tanh(H W₂ᵀ) w  ∈ ℝᴺ           │
 │  aᵢ = softmax(eᵢ)                   │
 │  m = Σ aᵢ hᵢ  ∈ ℝᴸ  (bag feature)  │
 └──────────────────────────────────────┘
         │
         ├── concat clinical features (5 vars × 10 expansion = 50-dim)
         │
         ▼  fused feature ∈ ℝ³⁰⁶
         │
         ├──► Metastasis Head  Linear → 64 → 2    (Binary CE loss, weight λ₁)
         │
         └──► Stage Head       Linear → 64 → 3    (3-class CE loss, weight λ₂)

Total loss = λ₁ · L_metastasis + λ₂ · L_stage  +  λ_wd · ‖θ‖²
```

The attention module learns to up-weight diagnostically relevant patches without requiring patch-level labels.

---

## Results

### Test Set Performance (Patient-Level Aggregation)

| Model | Feature Extractor | Metas. Acc | Metas. AUC | Metas. F1 | Sensitivity | Specificity | PPV | NPV | Stage Acc | Stage AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (single-task) | VGG16-BN (unfrozen) | 0.756 | 0.831 | — | **0.892** | 0.671 | 0.630 | **0.900** | — | — |
| **Multi-task (proposed)** | **UNI (frozen)** | 0.752 | **0.839** | **0.730** | 0.869 | 0.679 | 0.629 | 0.892 | **0.665** | **0.790** |
| Multi-task | VGG16-BN (unfrozen) | 0.633 | 0.714 | 0.375 | 0.286 | 0.851 | 0.545 | 0.655 | 0.583 | 0.672 |
| Single-task (metastasis) | UNI (frozen) | 0.734 | 0.820 | 0.707 | 0.833 | 0.672 | 0.614 | 0.865 | — | — |
| Single-task (stage only) | UNI (frozen) | — | — | — | — | — | — | — | 0.665 | 0.784 |

### Training & Validation AUC

| Model | Feature Extractor | Train AUC | Train Acc | Val AUC | Val Acc |
|---|---|:---:|:---:|:---:|:---:|
| Baseline | VGG16-BN | 0.88 | 0.77 | 0.82 | 0.76 |
| Multi-task (proposed) | UNI | 0.97 | 0.94 | **0.91** | **0.82** |
| Multi-task | VGG16-BN | 0.94 | 0.87 | 0.83 | 0.78 |
| Single-task (metastasis) | UNI | 0.98 | 0.93 | 0.91 | 0.81 |

---

## Key Findings

**UNI outperforms VGG16-BN across all models.** The pathology foundation model consistently yields higher AUC and better generalization, confirming the value of domain-specific pretraining for computational pathology tasks.

**Multi-task learning with UNI achieves best overall AUC (0.839).** Compared to the single-task UNI baseline (AUC 0.820), multi-task learning provides marginal but consistent improvements of 1–4% across nearly all metastasis metrics, with the added benefit of predicting ALN stage simultaneously (Stage AUC 0.790 vs. 0.784 for single-task stage model).

**Clinical screening profile.** All UNI-based models share a clinically favorable pattern — high sensitivity (>0.83) and high NPV (0.87–0.89) — meaning very few metastasis cases are missed. This is the correct tradeoff for oncological screening, where false negatives carry far greater clinical consequences than false positives.

**Data splitting matters.** The baseline's patient-level stratification created uneven bag-level class distribution across splits (35% positive bags in training vs. 45% in test), making direct bag-level comparison misleading. Patient-level aggregation at inference (mean probability for metastasis, majority vote for stage) substantially improves all metrics and is the reported evaluation protocol.

---

## Model Variants

| Model | Clinical Data | Architecture | Train Script |
|---|---|---|---|
| Baseline | Yes | Single-task metastasis, VGG16-BN | `train_singletask.py` |
| Single-task metastasis | Yes | Single-task, UNI or VGG16-BN | `train_singletask.py` |
| Single-task stage | Yes | Single-task 3-class, UNI | `train_singletask_status.py` |
| Multi-task (separate heads) | Yes | Dual-head, no shared trunk | `train_multitask.py` |
| Multi-task (shared trunk) | Yes | Shared FC layers → dual heads | `train_multitask.py --shared_layer` |
| Multi-task (image only) | No | Dual-head, no clinical fusion | `train_multitask.py --image_only` |

---

## Repository Structure

```
code/
├── mil_net.py                     # All model definitions
├── backbone_builder.py            # VGG16-BN and UNI backbone wrappers
├── attention_aggregator.py        # Attention-based MIL pooling (ABMIL)
├── dataset_loader.py              # PyTorch Dataset: patches + clinical data
├── train_multitask.py             # Multi-task training loop
├── train_singletask.py            # Single-task metastasis training
├── train_singletask_status.py     # Single-task stage training
├── update_status_data.ipynb       # Label preprocessing notebook
├── requirements.txt
└── dataset/
    ├── json/                      # Bag definitions (patch paths + labels)
    └── clinical_data/             # Preprocessed clinical feature spreadsheets
evaluation/
├── evaluate_multitask.py          # Multi-task evaluation (bag + patient level)
├── evaluate_singletask.py         # Single-task metastasis evaluation
└── evaluate_singletask_status.py  # Single-task stage evaluation
report files/
├── main.tex                       # Full project report (LaTeX)
└── name.tex
```

---

## Setup

```bash
pip install -r code/requirements.txt
```

UNI requires a Hugging Face token with access to `MahmoodLab/UNI2-h`. Authenticate via:
```bash
huggingface-cli login
```

---

## Training

**Multi-task model with UNI backbone (best configuration):**
```bash
cd code
python train_multitask.py \
  --data_dir_path /path/to/patches \
  --log_dir_path /path/to/logs \
  --log_name multitask_uni \
  --backbone uni2-h-freeze \
  --lr 1e-4 \
  --dropout 0.5 \
  --weight_decay 0.03 \
  --epoch 200
```

**Single-task metastasis baseline:**
```bash
python train_singletask.py \
  --data_dir_path /path/to/patches \
  --log_dir_path /path/to/logs \
  --log_name singletask_uni \
  --backbone uni2-h-freeze \
  --lr 1e-4 \
  --dropout 0.2 \
  --weight_decay 0.015
```

**Resume from checkpoint:**
```bash
python train_multitask.py \
  --resume_path /path/to/checkpoint/epoch_50.pth \
  ... # same args as original run
```

---

## Evaluation

```bash
cd code
python evaluation/evaluate_multitask.py \
  --checkpoint_path /path/to/best_combined.pth \
  --data_dir_path /path/to/patches \
  --backbone uni2-h-freeze \
  --csv_log_path results.csv \
  --model_name multitask_uni
```

Results are logged to CSV at both bag level and patient level automatically.

---

## Hyperparameter Tuning

| Hyperparameter | Search Space | Best Value |
|---|---|---|
| Metastasis loss weight λ₁ | {1, 2, 3} | 1 |
| Weight decay λ_wd | {0.015, 0.02, 0.03} | 0.03 (UNI multi-task) |
| Dropout rate | {0.2, 0.3, 0.4, 0.5} | 0.5 (UNI multi-task) |
| Learning rate | fixed | 1e-4 |

Weight decay and dropout had the most pronounced effect on generalization. Loss weight had minimal impact.

---

## Technical Details

| Component | Detail |
|---|---|
| MIL pooling | Attention-based weighted sum (Xu et al. 2021) |
| UNI output dim | 1,536 (ViT-Giant, patch 14, depth 24, 24 heads) |
| VGG16-BN output dim | 25,088 (512 × 7 × 7) |
| Attention projection | Linear(D→256) → FC(256→1) → Softmax |
| Clinical features | 5 variables, tiled ×10 → 50-dim for scale matching |
| Optimizer | Adam with L2 weight decay |
| LR scheduler | Cosine annealing warm restarts (T₀=20) |
| Early stopping | Train AUC ≥ 0.98 |
| Checkpointing | Best combined AUC + best metastasis AUC + every N epochs |
| Patient-level inference | Mean probability (metastasis) / majority vote (stage) |
| Monitoring | TensorBoard (loss + AUC per task per split) |

---

## Future Work

- **Task-specific attention mechanisms** to better capture stage-specific morphological features
- **Progressive layer sharing** or **dynamic task weighting** to maximize positive transfer
- **Fine-tuning UNI** (unfrozen) with careful learning rate scheduling
- **Patient-level model selection** to match evaluation protocol and close the validation–test gap

---

## References

1. Xu H. et al. "When multiple instance learning meets foundation models." *Medical Image Analysis* (2025).
2. Park D. et al. "Multimodal AI model for preoperative prediction of ALN metastasis." *npj Precision Oncology* (2025).
3. Xu F. et al. "Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning." *Frontiers in Oncology* (2021). [[GitHub](https://github.com/bupt-ai-cz/BALNMP)]
4. BCNB Dataset — [bcnb.grand-challenge.org](https://bcnb.grand-challenge.org/)
5. Chen R.J. et al. "Towards a general-purpose foundation model for computational pathology." *Nature Medicine* (2024).
