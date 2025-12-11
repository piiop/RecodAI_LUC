# Recod/LUC 2025 — Local Training + Kaggle Evaluation Pipeline

This repository contains the full local experimentation workflow for the Recod/LUC 2025 Kaggle competition.  
All model development, ablations, and tuning are performed **locally** using **OOF + CV metrics**.  
Final candidate models are then trained on **full data**, saved, uploaded to Kaggle, and evaluated via the **Kaggle LB** to guide daily submission strategy.

The workflow supports:
- Efficient local CV experimentation  
- Overnight full-train runs with W&B tracking  
- Easy ablation testing  
- Quick submission generation through a single Kaggle notebook  

---

## Directory Structure

```text
project_root/
├── README.md
├── prompts.txt
├── config/
│   ├── base.yaml
│   ├── best_oof.yaml
│   ├── ablations/
│   │   ├── ablation_lr.yaml
│   │   ├── ablation_feats.yaml
│   │   └── ...
│   └── kaggle_submission_presets.yaml
├── src/
│   ├── data/
│   │   └── dataloader.py
│   ├── models/
│   │   ├── mask2former_v1.py
│   │   ├── kaggle_metric.py
│   │   └── losses_metrics.py
│   ├── training/
│   │   ├── train_full.py
│   │   ├── train_cv.py
│   │   └── utils_trainer.py
│   ├── inference/
│   │   ├── infer_multi_configs.py
│   │   └── postprocess.py
│   ├── utils/
│   │   ├── wandb_utils.py
│   │   ├── config_utils.py
│   │   └── seed_logging_utils.py
│   └── cli.py
├── experiments/
│   ├── oof_results/
│   ├── wandb/
│   └── ablation_plans.md
├── weights/
│   ├── local_oof/
│   ├── full_train/
│   └── kaggle_artifacts/
├── kaggle/
│   └── submission_notebook.ipynb
└── notebooks/
    ├── EDA.ipynb
    ├── baseline_prototype.ipynb
    ├── cv_analysis.ipynb
    ├── kagglelb_analysis.ipynb
    └── debugger.ipynb
Overview

This structure allows:

Local OOF + CV Development

train_cv.py produces robust local metrics.

Scores guide hyperparameter tuning and ablation experiments.

Full-Data Training

train_full.py uses the best OOF-configured settings.

Outputs final weights for Kaggle submission evaluation.

Submission Testing

Weights are uploaded manually to Kaggle.

submission_notebook.ipynb generates predictions + submission files.

Up to 5/day LB scores are used to validate promising models.

Ablations + Daily LB Strategy

infer_multi_configs.py helps queue a set of candidate configs for LB testing.

Ablation results are tracked with W&B and stored under experiments/.