#!/usr/bin/env bash
set -euo pipefail

CFG=base_v2.yaml
EPOCHS=5

# bash src/training/new_loss_test2.sh

#!/usr/bin/env bash
set -euo pipefail

CFG="${CFG:-base_v2.yaml}"
EPOCHS="${EPOCHS:-5}"

# Notes (training knobs in current code):
# - compute_losses supports: loss_weight_mask_{bce,dice,cls}, loss_weight_presence,
#   loss_weight_auth_penalty, authenticity_penalty_weight :contentReference[oaicite:0]{index=0}
# - model exposes authenticity_penalty_weight and loss_weight_auth_penalty (and other ctor cfg) :contentReference[oaicite:1]{index=1}
# - src.cli forwards -o dotted overrides into model kwargs :contentReference[oaicite:2]{index=2}

# 1) Baseline (new loss system) — just name it clearly
python -m src.cli cv -c "$CFG" \
  -o trainer.name=mini_lossv2_baseline \
  -o trainer.epochs="$EPOCHS"

# 2) Much stronger authentic suppression (forces sparsity by punishing any activity on authentic)
python -m src.cli cv -c "$CFG" \
  -o trainer.name=mini_lossv2_authpen_strong20 \
  -o trainer.epochs="$EPOCHS" \
  -o model.authenticity_penalty_weight=20.0

# 3) Weak authentic suppression (should allow activity; useful to confirm penalty is actually doing something)
python -m src.cli cv -c "$CFG" \
  -o trainer.name=mini_lossv2_authpen_weak1 \
  -o trainer.epochs="$EPOCHS" \
  -o model.authenticity_penalty_weight=1.0

# 4) Push “how many instances” harder (sparser queries via stronger query-classification pressure)
python -m src.cli cv -c "$CFG" \
  -o trainer.name=mini_lossv2_cls_up2 \
  -o trainer.epochs="$EPOCHS" \
  -o model.loss_weight_mask_cls=2.0

# 5) Reduce query capacity (fewer queries => easier to concentrate signal + less noise)
python -m src.cli cv -c "$CFG" \
  -o trainer.name=mini_lossv2_numq8_authpen10 \
  -o trainer.epochs="$EPOCHS" \
  -o model.num_queries=8 \
  -o model.authenticity_penalty_weight=10.0

