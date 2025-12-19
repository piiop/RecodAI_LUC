#!/usr/bin/env bash
set -euo pipefail

# mini CV sweep: 5 epochs, 3 folds
# run: bash scripts/mini_cv_sweep.sh
# bash src/training/new_loss_tests.sh



CFG=base_v2.yaml
EPOCHS=5

# 1) Presence gate only (baseline)
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_presence_0p5 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_presence_threshold=0.5

# 2) Presence gate (looser)
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_presence_0p3 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_presence_threshold=0.3

# 3) Query-score filter only
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_qscore_0p5 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_qscore_threshold=0.5

# 4) Top-K only
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_topk_1 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_topk=1

# 5) Min mask mass only
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_minmass_0p002 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_min_mask_mass=0.002

