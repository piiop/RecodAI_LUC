#!/usr/bin/env bash
set -euo pipefail

CFG="${CFG:-base_v2.yaml}"
EPOCHS="${EPOCHS:-5}"

# Near-current-best baseline knobs:
# - loss_weight_mask_cls=2.0
# - authenticity_penalty_weight=1.0
# Keep train_topk/train_min_mask_mass as in base_v2 unless overridden.
COMMON=(
  -c "$CFG"
  -o trainer.epochs="$EPOCHS"
  -o model.loss_weight_mask_cls=2.0
  -o model.authenticity_penalty_weight=1.0
)

# 1) No TV (control)
python -m src.cli cv "${COMMON[@]}" \
  -o trainer.name=mini_tv_0_control \
  -o model.tv_lambda=0.0

# 2) Light TV (below current default)
python -m src.cli cv "${COMMON[@]}" \
  -o trainer.name=mini_tv_0p02 \
  -o model.tv_lambda=0.02

# 3) Current default TV (as in base_v2.yaml)
python -m src.cli cv "${COMMON[@]}" \
  -o trainer.name=mini_tv_0p05_default \
  -o model.tv_lambda=0.05

# 4) Strong TV (stress test)
python -m src.cli cv "${COMMON[@]}" \
  -o trainer.name=mini_tv_0p10_strong \
  -o model.tv_lambda=0.10
