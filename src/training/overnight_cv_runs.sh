#!/usr/bin/env bash
set -euo pipefail

# Notes (why these runs):
# - CV uses Mask2FormerForgeryModel(**model_kwargs) from CLI overrides :contentReference[oaicite:0]{index=0} and trains via run_cv :contentReference[oaicite:1]{index=1}.
# - Your *training* behavior is dominated by: (a) Hungarian matching costs (cost_bce/cost_dice), (b) mask loss weights,
#   (c) mask-cls loss weight, and (d) authenticity-penalty settings (weight + thr + temperature) :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}.
# - Inference thresholds are already effectively off in this project setup :contentReference[oaicite:4]{index=4}.

# 1) Matching-cost ablation: Dice-dominant matching (does the matcher align better on shapes, helping masks/cls learn?)
python -m src.cli cv -c base.yaml \
  -o trainer.name=match_dice_dominant_cost2_bce0p5 \
  -o trainer.epochs=20 \
  -o model.cost_bce=0.5 \
  -o model.cost_dice=2.0 \
  -o model.loss_weight_mask_bce=1.0 \
  -o model.loss_weight_mask_dice=1.0 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.loss_weight_img_auth=1.0 \
  -o model.loss_weight_auth_penalty=1.0 \
  -o model.authenticity_penalty_weight=3.0 \
  -o model.auth_penalty_cls_threshold=0.30 \
  -o model.auth_penalty_temperature=0.30

# 2) Matching-cost ablation: BCE-dominant matching (tests whether pixelwise “easiness” matching avoids bad early assignments)
python -m src.cli cv -c base.yaml \
  -o trainer.name=match_bce_dominant_cost2_dice0p5 \
  -o trainer.epochs=20 \
  -o model.cost_bce=2.0 \
  -o model.cost_dice=0.5 \
  -o model.loss_weight_mask_bce=1.0 \
  -o model.loss_weight_mask_dice=1.0 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.loss_weight_img_auth=1.0 \
  -o model.loss_weight_auth_penalty=1.0 \
  -o model.authenticity_penalty_weight=3.0 \
  -o model.auth_penalty_cls_threshold=0.30 \
  -o model.auth_penalty_temperature=0.30

# 3) Mask-loss ratio ablation: Dice-heavy loss (if masks are “blobby/overconfident”, emphasize overlap quality)
python -m src.cli cv -c base.yaml \
  -o trainer.name=loss_dice_heavy_bce0p5_dice2 \
  -o trainer.epochs=20 \
  -o model.loss_weight_mask_bce=0.5 \
  -o model.loss_weight_mask_dice=2.0 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.loss_weight_img_auth=1.0 \
  -o model.loss_weight_auth_penalty=1.0 \
  -o model.authenticity_penalty_weight=3.0 \
  -o model.auth_penalty_cls_threshold=0.30 \
  -o model.auth_penalty_temperature=0.30

# 4) “Penalty shape” ablation: very soft penalty gate (high temp) + moderate weight (should discourage FP on authentic without killing cls head)
python -m src.cli cv -c base.yaml \
  -o trainer.name=soft_auth_penalty_temp1_thr0p2_w3 \
  -o trainer.epochs=20 \
  -o model.loss_weight_mask_bce=1.0 \
  -o model.loss_weight_mask_dice=1.0 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.loss_weight_img_auth=1.0 \
  -o model.loss_weight_auth_penalty=1.0 \
  -o model.authenticity_penalty_weight=3.0 \
  -o model.auth_penalty_cls_threshold=0.20 \
  -o model.auth_penalty_temperature=1.00

# 5) “Penalty schedule” proxy: delayed/stricter gate (higher thr, lower temp) but smaller weight (tests if sharp penalty is the collapse trigger)
python -m src.cli cv -c base.yaml \
  -o trainer.name=sharp_auth_penalty_temp0p05_thr0p6_w2 \
  -o trainer.epochs=20 \
  -o model.loss_weight_mask_bce=1.0 \
  -o model.loss_weight_mask_dice=1.0 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.loss_weight_img_auth=1.0 \
  -o model.loss_weight_auth_penalty=1.0 \
  -o model.authenticity_penalty_weight=2.0 \
  -o model.auth_penalty_cls_threshold=0.60 \
  -o model.auth_penalty_temperature=0.05
