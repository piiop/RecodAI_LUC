#!/usr/bin/env bash
set -e  # stop if any command fails

python -m src.cli cv -c base.yaml \
  -o trainer.name=mask_first_wmask4_wcls0p25 \
  -o trainer.epochs=20 \
  -o model.loss_weight_mask_bce=2.0 \
  -o model.loss_weight_mask_dice=2.0 \
  -o model.loss_weight_mask_cls=0.25 \
  -o model.loss_weight_img_auth=1.0 \
  -o model.loss_weight_auth_penalty=0.5 \
  -o model.authenticity_penalty_weight=2.0

python -m src.cli cv -c base.yaml \
  -o trainer.name=balanced_soft_penalty_thr0p3_temp0p3 \
  -o trainer.epochs=20 \
  -o model.loss_weight_mask_bce=1.0 \
  -o model.loss_weight_mask_dice=1.0 \
  -o model.loss_weight_mask_cls=1.0 \
  -o model.loss_weight_img_auth=1.0 \
  -o model.loss_weight_auth_penalty=1.0 \
  -o model.authenticity_penalty_weight=3.0 \
  -o model.auth_penalty_cls_threshold=0.30 \
  -o model.auth_penalty_temperature=0.30

python -m src.cli cv -c base.yaml \
  -o trainer.name=anti_collapse_wcls2_strong_auth_penalty \
  -o trainer.epochs=20 \
  -o model.loss_weight_mask_bce=1.0 \
  -o model.loss_weight_mask_dice=1.0 \
  -o model.loss_weight_mask_cls=2.0 \
  -o model.loss_weight_img_auth=1.0 \
  -o model.loss_weight_auth_penalty=1.0 \
  -o model.authenticity_penalty_weight=6.0 \
  -o model.auth_penalty_cls_threshold=0.50 \
  -o model.auth_penalty_temperature=0.10

python -m src.cli cv -c base.yaml \
  -o trainer.name=cls_unclamp_test \
  -o trainer.epochs=20 \
  -o model.loss_weight_mask_bce=1.0 \
  -o model.loss_weight_mask_dice=1.0 \
  -o model.loss_weight_mask_cls=0.05 \
  -o model.loss_weight_img_auth=1.0 \
  -o model.loss_weight_auth_penalty=0.0 \
  -o model.authenticity_penalty_weight=0.0
