#!/usr/bin/env bash
set -euo pipefail

# bash src/training/new_full_runs.sh

CFG="${CFG:-base_v2.yaml}"
EPOCHS_OOF="${EPOCHS_OOF:-25}"

COMMON=(
  -c "$CFG"
  -o trainer.epochs="$EPOCHS_OOF"
  -o model.loss_weight_mask_cls=2.0
  -o model.authenticity_penalty_weight=1.0
)

# Strong TV (stress test) + convnext_base
python -m src.cli cv "${COMMON[@]}" \
  -o trainer.name=mini_tv_0p10_strong_cnb \
  -o model.tv_lambda=0.10 \
  -o model.backbone_name=convnext_base

# Strong TV + reduced cls pressure (diagnostic) + convnext_base
python -m src.cli cv "${COMMON[@]}" \
  -o trainer.name=mini_tv_0p10_cls1p0_cnb \
  -o model.tv_lambda=0.10 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.backbone_name=convnext_base