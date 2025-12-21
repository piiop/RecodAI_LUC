#!/usr/bin/env bash
set -euo pipefail

CFG="${CFG:-base_v2.yaml}"
EPOCHS_OOF="${EPOCHS_OOF:-25}"

# bash src/training/new_full_runs.sh

# 1) Best mini mean: push cls loss
python -m src.cli cv -c "$CFG" \
  -o trainer.name=oof_lossv2_cls_up2_e${EPOCHS_OOF} \
  -o trainer.epochs="$EPOCHS_OOF" \
  -o model.loss_weight_mask_cls=2.0

# 2) Runner-up: weaker auth penalty (less “all-authentic” pressure)
python -m src.cli cv -c "$CFG" \
  -o trainer.name=oof_lossv2_authpen_weak1_e${EPOCHS_OOF} \
  -o trainer.epochs="$EPOCHS_OOF" \
  -o model.authenticity_penalty_weight=1.0

# 3) Combine the two best knobs (if sparsity is truly training-driven, this should win)
python -m src.cli cv -c "$CFG" \
  -o trainer.name=oof_lossv2_cls_up2_authpen_weak1_e${EPOCHS_OOF} \
  -o trainer.epochs="$EPOCHS_OOF" \
  -o model.loss_weight_mask_cls=2.0 \
  -o model.authenticity_penalty_weight=1.0
