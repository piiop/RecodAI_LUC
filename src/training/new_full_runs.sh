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
  -o trainer.n_folds=3 \
  -o trainer.name=oof_tv_0p10_cls1p0_cnb_tk1_mm0p002_pr0p02_m0p5_img512 \
  -o data.img_size=512 \
  -o model.tv_lambda=0.10 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.authenticity_penalty_weight=1.0 \
  -o model.backbone_name=convnext_base \
  -o model.default_topk=1 \
  -o model.default_qscore_threshold=null \
  -o model.default_min_mask_mass=0.002 \
  -o model.default_presence_threshold=0.02 \
  -o model.default_mask_threshold=0.5

# Strong TV + reduced cls pressure (diagnostic) + convnext_base
python -m src.cli cv "${COMMON[@]}" \
  -o trainer.n_folds=3 \
  -o trainer.name=oof_tv_0p10_cls1p0_cnb_tk1_mm0p002_pr0p02_m0p5_img256a \
  -o data.img_size=256 \
  -o model.tv_lambda=0.10 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.authenticity_penalty_weight=1.0 \
  -o model.backbone_name=convnext_base \
  -o model.default_topk=1 \
  -o model.default_qscore_threshold=null \
  -o model.default_min_mask_mass=0.002 \
  -o model.default_presence_threshold=0.02 \
  -o model.default_mask_threshold=0.5

  python -m src.cli cv "${COMMON[@]}" \
  -o trainer.n_folds=3 \
  -o trainer.name=oof_tv_0p10_cls1p0_cnb_tk1_mm0p002_pr0p02_m0p5_img256b \
  -o data.img_size=256 \
  -o model.tv_lambda=0.10 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.authenticity_penalty_weight=1.0 \
  -o model.backbone_name=convnext_base \
  -o model.default_topk=1 \
  -o model.default_qscore_threshold=null \
  -o model.default_min_mask_mass=0.002 \
  -o model.default_presence_threshold=0.02 \
  -o model.default_mask_threshold=0.5