# =============================================================================
# Recod/LUC 2025 – Local Training & Sweeps Makefile
#
# Common commands:
#   make cv-mini        # quick CV smoke test
#   make cv             # full CV run (default settings)
#   make full-baseline  # full-data training, baseline weights
#   make gate-sweep     # sweep authenticity gate on full model
#   make multi-configs  # run infer_multi_configs presets
# =============================================================================

# --- Global config -----------------------------------------------------------

PYTHON              ?= python
TRAIN_AUTHENTIC     ?= data/train_images/authentic
TRAIN_FORGED        ?= data/train_images/forged
TRAIN_MASKS         ?= data/train_masks
SUPP_FORGED         ?=
SUPP_MASKS          ?=

# Where experiments & weights live (matches README layout) 
EXPER_OOF_DIR       ?= experiments/oof_results
EXPER_MULTI_DIR     ?= experiments/multi_configs
EXPER_GATE_DIR      ?= experiments/auth_gate_sweep
WEIGHTS_FULL_DIR    ?= weights/full_train
FULL_BASELINE_PTH   ?= $(WEIGHTS_FULL_DIR)/model_full_data_baseline.pth

# Wandb can be disabled via env if desired:
#   WANDB_DISABLED=true make cv
# -----------------------------------------------------------------------------


.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make cv-mini        - quick CV smoke test (few epochs, small out_dir)"
	@echo "  make cv             - full CV run with defaults"
	@echo "  make full-baseline  - full-data training, baseline weights"
	@echo "  make gate-sweep     - sweep auth gate thresholds on training set"
	@echo "  make multi-configs  - run infer_multi_configs presets"
	@echo
	@echo "Config via env:"
	@echo "  TRAIN_AUTHENTIC, TRAIN_FORGED, TRAIN_MASKS, SUPP_FORGED, SUPP_MASKS"
	@echo "  EXPER_OOF_DIR, EXPER_MULTI_DIR, EXPER_GATE_DIR, FULL_BASELINE_PTH"


# =============================================================================
# CV runs (via src.cli cv -> src/training/train_cv.py) 
# =============================================================================

.PHONY: cv-mini
cv-mini:
	$(PYTHON) -m src.cli cv \
	  --train_authentic $(TRAIN_AUTHENTIC) \
	  --train_forged   $(TRAIN_FORGED) \
	  --train_masks    $(TRAIN_MASKS) \
	  $(if $(SUPP_FORGED),--supp_forged $(SUPP_FORGED),) \
	  $(if $(SUPP_MASKS),--supp_masks $(SUPP_MASKS),) \
	  --n_folds 3 \
	  --epochs 1 \
	  --batch_size 2 \
	  --lr 1e-4 \
	  --weight_decay 1e-4 \
	  --out_dir $(EXPER_OOF_DIR)/mini_smoke

.PHONY: cv
cv:
	$(PYTHON) -m src.cli cv \
	  --train_authentic $(TRAIN_AUTHENTIC) \
	  --train_forged   $(TRAIN_FORGED) \
	  --train_masks    $(TRAIN_MASKS) \
	  $(if $(SUPP_FORGED),--supp_forged $(SUPP_FORGED),) \
	  $(if $(SUPP_MASKS),--supp_masks $(SUPP_MASKS),) \
	  --n_folds 5 \
	  --epochs 3 \
	  --batch_size 4 \
	  --lr 1e-4 \
	  --weight_decay 1e-4 \
	  --out_dir $(EXPER_OOF_DIR)/baseline


# =============================================================================
# Full-data training (via src.cli full-train -> src/training/train_full.py) 
# =============================================================================

.PHONY: full-baseline
full-baseline:
	$(PYTHON) -m src.cli full-train \
	  --train_authentic $(TRAIN_AUTHENTIC) \
	  --train_forged   $(TRAIN_FORGED) \
	  --train_masks    $(TRAIN_MASKS) \
	  $(if $(SUPP_FORGED),--supp_forged $(SUPP_FORGED),) \
	  $(if $(SUPP_MASKS),--supp_masks $(SUPP_MASKS),) \
	  --epochs 30 \
	  --batch_size 4 \
	  --lr 1e-4 \
	  --weight_decay 1e-4 \
	  --save_path $(FULL_BASELINE_PTH)


# =============================================================================
# Authenticity gate sweep (auth_gate_sweep.py you added under src/inference/)
# Uses full-baseline weights; computes Kaggle metric on training set.
# =============================================================================

.PHONY: gate-sweep
gate-sweep:
	$(PYTHON) -m src.inference.auth_gate_sweep \
	  --train_authentic $(TRAIN_AUTHENTIC) \
	  --train_forged   $(TRAIN_FORGED) \
	  --train_masks    $(TRAIN_MASKS) \
	  $(if $(SUPP_FORGED),--supp_forged $(SUPP_FORGED),) \
	  $(if $(SUPP_MASKS),--supp_masks $(SUPP_MASKS),) \
	  --weights $(FULL_BASELINE_PTH) \
	  --gates 0.3 0.4 0.5 0.6 0.7 \
	  --batch_size 4 \
	  --out_dir $(EXPER_GATE_DIR)


# =============================================================================
# Multi-config CV experiments (src/inference/infer_multi_configs.py) 
# =============================================================================

.PHONY: multi-configs
multi-configs:
	$(PYTHON) -m src.inference.infer_multi_configs \
	  --train_authentic $(TRAIN_AUTHENTIC) \
	  --train_forged   $(TRAIN_FORGED) \
	  --train_masks    $(TRAIN_MASKS) \
	  $(if $(SUPP_FORGED),--supp_forged $(SUPP_FORGED),) \
	  $(if $(SUPP_MASKS),--supp_masks $(SUPP_MASKS),) \
	  --base_out_dir $(EXPER_MULTI_DIR)


# =============================================================================
# Utility targets
# =============================================================================

.PHONY: clean-oof
clean-oof:
	rm -rf $(EXPER_OOF_DIR)

.PHONY: clean-multi
clean-multi:
	rm -rf $(EXPER_MULTI_DIR)

.PHONY: clean-gate
clean-gate:
	rm -rf $(EXPER_GATE_DIR)
