# Ablation Plans

This document outlines early-stage and upcoming ablation ideas for developing a strong solution for the Recod/LUC 2025 forgery-segmentation competition.  
Each ablation should be run through the local **OOF CV pipeline** and optionally validated through a **full-data model → Kaggle LB check**.

---

## 1. Data Augmentation Ablations (Early Phase)

### 1.1 Augmentations OFF (baseline sanity)
- Purpose: confirm model learns without augmentation noise.
- Config change: minimal transform → resize + normalize only.
- Expected: lower generalization; useful as a clean baseline comparison.

### 1.2 Light Augmentations ON
- Horizontal flips  
- Color jitter (small)  
- Mild geometric transforms  

### 1.3 Strong Augmentations
- Cutout / coarse dropout  
- Random elastic distortions (controlled)  
- Random resized crops  
- Expectation: may help robustness for irregular forgery masks.

---

## 2. Backbone Ablations

### 2.1 convnext_tiny (current default)
- Baseline reference.

### 2.2 convnext_small / base
- Higher capacity; test diminishing returns vs training time.

### 2.3 Swin-T / Swin-S
- Self-attention spatial modeling; often strong for segmentation.

### 2.4 EfficientViT / MobileViT style small backbones
- Check if lightweight architectures remain competitive.

---

## 3. Decoder / Query Ablations

### 3.1 Number of queries
- 10 vs 20 vs 30 queries.
- Affects ability to capture multiple small forgery regions.

### 3.2 Freeze vs train backbone
- Freeze early blocks to test dependency on feature adaptation.

---

## 4. Post-Processing Ablations

### 4.1 Mask binarization threshold
- Test thresholds 0.4 / 0.5 / 0.6.

### 4.2 Instance merging logic
- Merge masks if IoU > X.
- Useful when forgeries produce multiple small fragments.

### 4.3 Small artifact removal
- Remove components below Y pixels or below % of image area.

### 4.4 Edge sharpening / CRF
- Optional refinement for boundary fidelity.

---

## 5. Loss & Training Signal Ablations

### 5.1 Authenticity penalty weight
- Tune penalty for false-positive forged predictions on authentic images.

### 5.2 BCE vs BCE+DICE weighting
- Explore stronger dice emphasis for thin or irregular regions.

### 5.3 Mask-classification loss temperature / focal settings
- Reduce overconfident wrong region predictions.

---

## 6. Forgery-Specific Ablations (Recommended)

### 6.1 Boundary-aware losses
- Add boundary dice or gradient loss to better capture contour irregularity.

### 6.2 Contrastive patch embeddings
- Encourage forged vs authentic patches to separate in embedding space.

### 6.3 Texture discrepancy modeling
- Simple ablation: extra channel input = Laplacian / Sobel maps.

### 6.4 Multi-resolution inference
- Predictions at 256 → 384 → 512; average or merge masks.

---

## 7. Inference Strategy Ablations

### 7.1 TTA ON vs OFF
- Horizontal flip, multi-scale.

### 7.2 Multi-model ensembling
- Average masks from best configurations.

---

## Usage Notes
- Each ablation should have a corresponding YAML override under `config/ablations/`.
- Keep modifications isolated and minimal so CV score movements are easy to interpret.
- After promising changes, run **full-data training** and test on **Kaggle LB**.

