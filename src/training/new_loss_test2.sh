#!/usr/bin/env bash
set -euo pipefail

CFG=base_v2.yaml
EPOCHS=5

# bash src/training/new_loss_test2s.sh

# Why these 5:
# (A) Your results show the *only* non-degenerate improvements came from "conservative instance selection"
#     (topk/min_mass). So we bias runs toward: fewer queries + fewer tiny masks.
# (B) Presence gating looked brittle (fold flipping). We test it only in “soft” combos where it
#     suppresses obvious garbage without forcing all-auth collapse.
# (C) We include 1 "pretty good" conservative baseline + 4 “extreme-ish” variants around it.

# 1) "Pretty good" conservative baseline:
#    topk=1 (strong prior: single best query) + tiny min_mass to drop speckle.
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_topk1_minmass_0p001 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_topk=1 \
  -o model.default_min_mask_mass=0.001

# 2) Extreme: topk=1 + stricter min_mass to force only chunky masks through.
#    Tests if your partial gains were coming from removing micro-FPs.
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_topk1_minmass_0p003 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_topk=1 \
  -o model.default_min_mask_mass=0.003

# 3) Extreme: topk=2 (allow 2 instances) + min_mass=0.002.
#    Checks whether some folds truly need 2 instances, while still filtering speckle.
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_topk2_minmass_0p002 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_topk=2 \
  -o model.default_min_mask_mass=0.002

# 4) Extreme: qscore threshold only (no topk), tuned low-ish to avoid all-auth.
#    Tests whether "absolute activity" selection beats fixed instance count.
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_qscore_0p15_minmass_0p002 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_qscore_threshold=0.15 \
  -o model.default_min_mask_mass=0.002

# 5) Extreme: soft presence gate + topk=1.
#    Goal: suppress obvious authentic FPs without hard-collapsing to all-auth.
#    (Presence=0.45 is intentionally near the observed brittle region, but less harsh than 0.5.)
python -m src.cli cv -c $CFG \
  -o trainer.name=mini_presence_0p45_topk1 \
  -o trainer.epochs=$EPOCHS \
  -o model.default_presence_threshold=0.45 \
  -o model.default_topk=1
