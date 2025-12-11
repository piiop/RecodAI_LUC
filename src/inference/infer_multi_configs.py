# src/inference/infer_multi_configs.py

"""
Run multiple Mask2Former CV configs on the local training data and compare
their OOF scores.

- Uses src.training.train_cv.run_cv (official Kaggle metric on OOF).
- No Kaggle submissions are created.
- Configs are defined as in-code hyperparam presets (see EXPERIMENTS below).
- Each config writes its own OOF outputs under base_out_dir/<exp_name>/.

Typical usage:

    python -m src.inference.infer_multi_configs \
        --train_authentic /path/to/train_images/authentic \
        --train_forged   /path/to/train_images/forged \
        --train_masks    /path/to/train_masks \
        --supp_forged    /path/to/supplemental_images \
        --supp_masks     /path/to/supplemental_masks \
        --experiment baseline \
        --experiment ablation_lr

If you omit --experiment, all presets in EXPERIMENTS are run (optionally
capped by --max_experiments).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.training.train_cv import run_cv
from src.utils.seed_logging_utils import setup_seed, log_seed_info
from src.utils.wandb_utils import init_wandb_run, log_config, finish_run


# ---------------------------------------------------------------------------
# Experiment presets (edit this list to define your sweeps)
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    name: str

    # CV / training hyperparams
    num_folds: int = 5
    num_epochs: int = 3
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Optional description (purely for logging / readability)
    desc: str = ""


# Initial presets – extend/modify as you like.
EXPERIMENTS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="baseline",
        num_folds=5,
        num_epochs=3,
        batch_size=4,
        lr=1e-4,
        weight_decay=1e-4,
        desc="Baseline settings matching train_cv defaults",
    ),
    # Example ablations (tweak or remove as needed)
    ExperimentConfig(
        name="lr_3e-4",
        num_folds=5,
        num_epochs=3,
        batch_size=4,
        lr=3e-4,
        weight_decay=1e-4,
        desc="Higher learning rate",
    ),
    ExperimentConfig(
        name="bs_8",
        num_folds=5,
        num_epochs=3,
        batch_size=8,
        lr=1e-4,
        weight_decay=1e-4,
        desc="Larger batch size",
    ),
]


def _index_experiments() -> Dict[str, ExperimentConfig]:
    return {exp.name: exp for exp in EXPERIMENTS}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_single_experiment(
    exp: ExperimentConfig,
    paths: Dict[str, str],
    device: torch.device,
    base_out_dir: Path,
) -> Dict[str, Optional[float]]:
    """
    Run one CV experiment and return its metrics (mean_cv, oof_score).
    """
    out_dir = base_out_dir / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Running experiment: {exp.name} =====")
    if exp.desc:
        print(f"  Description: {exp.desc}")
    print(f"  Output dir: {out_dir}")

    # Init a dedicated wandb run for this experiment (optional; no-op if wandb disabled)
    run = init_wandb_run(
        config={
            "experiment": exp.name,
            "num_folds": exp.num_folds,
            "num_epochs": exp.num_epochs,
            "batch_size": exp.batch_size,
            "lr": exp.lr,
            "weight_decay": exp.weight_decay,
            "paths": paths,
        },
        project="mask2former-forgery",
        job_type="cv",
        group="multi_configs",
        name=f"multi_{exp.name}",
    )

    try:
        run_cv(
            paths=paths,
            num_folds=exp.num_folds,
            num_epochs=exp.num_epochs,
            batch_size=exp.batch_size,
            lr=exp.lr,
            weight_decay=exp.weight_decay,
            device=device,
            train_transform=None,  # plug in your train/val transforms here when ready
            val_transform=None,
            out_dir=str(out_dir),
        )
    finally:
        finish_run()

    metrics_path = out_dir / "oof_metrics.json"
    if not metrics_path.is_file():
        print(f"[warn] No metrics file found at {metrics_path}")
        return {
            "name": exp.name,
            "mean_cv": None,
            "oof_score": None,
            "out_dir": str(out_dir),
        }

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    mean_cv = float(metrics.get("mean_cv")) if "mean_cv" in metrics else None
    oof_score = float(metrics.get("oof_score")) if "oof_score" in metrics else None

    print(
        f"  -> mean_cv={mean_cv:.6f}  oof_score={oof_score:.6f}"
        if mean_cv is not None and oof_score is not None
        else f"  -> metrics loaded from {metrics_path}"
    )

    return {
        "name": exp.name,
        "mean_cv": mean_cv,
        "oof_score": oof_score,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple Mask2Former CV configs on local data and compare OOF scores. "
            "No Kaggle submissions are produced."
        )
    )

    # Data paths (same shape as train_cv)
    parser.add_argument(
        "--train_authentic",
        type=str,
        required=True,
        help="Path to authentic train images directory",
    )
    parser.add_argument(
        "--train_forged",
        type=str,
        required=True,
        help="Path to forged train images directory",
    )
    parser.add_argument(
        "--train_masks",
        type=str,
        required=True,
        help="Path to train masks directory",
    )
    parser.add_argument(
        "--supp_forged",
        type=str,
        default=None,
        help="Path to supplemental forged images (optional)",
    )
    parser.add_argument(
        "--supp_masks",
        type=str,
        default=None,
        help="Path to supplemental masks (optional)",
    )

    # Experiment selection
    parser.add_argument(
        "--experiment",
        "-e",
        action="append",
        default=None,
        help=(
            "Name of an experiment preset to run (can repeat). "
            "If omitted, all presets in EXPERIMENTS are run."
        ),
    )
    parser.add_argument(
        "--max_experiments",
        type=int,
        default=None,
        help="Optional cap on number of experiments to run (after selection).",
    )
    parser.add_argument(
        "--base_out_dir",
        type=str,
        default="experiments/multi_configs",
        help="Root directory for per-experiment outputs.",
    )

    # Seed / determinism
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python/NumPy/PyTorch (default: 42)",
    )
    parser.add_argument(
        "--no_deterministic",
        action="store_false",
        dest="deterministic",
        help="Disable PyTorch deterministic mode (cuDNN, etc.)",
    )
    parser.set_defaults(deterministic=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths: Dict[str, str] = {
        "train_authentic": args.train_authentic,
        "train_forged": args.train_forged,
        "train_masks": args.train_masks,
    }
    if args.supp_forged is not None:
        paths["supp_forged"] = args.supp_forged
    if args.supp_masks is not None:
        paths["supp_masks"] = args.supp_masks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Seed & log environment once for the whole multi-run
    seed_info = setup_seed(args.seed, deterministic=args.deterministic)
    log_seed_info(seed_info)

    all_exps = _index_experiments()

    if args.experiment is None:
        selected_names = list(all_exps.keys())
    else:
        selected_names = []
        for name in args.experiment:
            if name not in all_exps:
                print(f"[warn] Unknown experiment '{name}' – skipping.")
                continue
            selected_names.append(name)

    if args.max_experiments is not None:
        selected_names = selected_names[: args.max_experiments]

    if not selected_names:
        print("No experiments selected – nothing to do.")
        return

    print("\nPlanned experiments:", ", ".join(selected_names))

    base_out_dir = Path(args.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Optional[float]]] = []
    for name in selected_names:
        exp = all_exps[name]
        res = run_single_experiment(exp, paths=paths, device=device, base_out_dir=base_out_dir)
        results.append(res)

    # Summary table
    print("\n===== Summary (sorted by mean_cv) =====")
    sortable = [r for r in results if r["mean_cv"] is not None]
    sortable.sort(key=lambda r: r["mean_cv"], reverse=True)

    for r in sortable:
        print(
            f"{r['name']:16s}  mean_cv={r['mean_cv']:.6f}  "
            f"oof_score={r['oof_score']:.6f}  out_dir={r['out_dir']}"
        )

    # Also list any runs that failed to produce metrics
    missing = [r for r in results if r["mean_cv"] is None]
    if missing:
        print("\nExperiments with missing metrics:")
        for r in missing:
            print(f"  {r['name']:16s}  (check {r['out_dir']})")


if __name__ == "__main__":
    main()
