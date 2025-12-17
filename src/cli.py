# src/cli.py
"""
Unified, simplified CLI for Mask2Former forgery experiments.

All configuration comes from YAML + CLI overrides.
CLI flags exist only for truly top-level concerns (command, device, seed).

Usage
-----
python -m src.cli cv -c base.yaml -o trainer.lr=3e-4
python -m src.cli full-train -c best_oof.yaml
"""

import argparse
from pathlib import Path

import torch

from src.training.train_cv import run_cv
from src.training.train_full import run_full_train
from src.utils.seed_logging_utils import setup_seed, log_seed_info
from src.utils.wandb_utils import init_wandb_run, finish_run
from src.utils.config_utils import add_config_arguments, build_config_from_args


# ---------------------------------------------------------------------
# Subparsers
# ---------------------------------------------------------------------

def _add_cv_subparser(subparsers):
    p = subparsers.add_parser("cv", help="Run K-fold CV and produce OOF results")
    add_config_arguments(p)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    p.set_defaults(deterministic=True)


def _add_full_train_subparser(subparsers):
    p = subparsers.add_parser("full-train", help="Train on full dataset and save weights")
    add_config_arguments(p)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    p.set_defaults(deterministic=True)


# ---------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------

def _run_cv(args):
    cfg = build_config_from_args(args)
    t = cfg.get("trainer", {})
    m = cfg.get("model", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed, deterministic=args.deterministic)

    run = init_wandb_run(
        config=cfg.to_dict(),
        project="mask2former-forgery",
        job_type="cv",
        group="cv",
        name=t.get("name", "cv"),
    )

    base_out_dir = Path(t.get("out_dir", "experiments/oof_results"))
    run_name = t.get("name", "cv")
    out_dir = base_out_dir / run_name

    try:
        run_cv(
            num_folds=t.get("n_folds", 5),
            num_epochs=t.get("epochs", 3),
            batch_size=t.get("batch_size", 4),
            lr=t.get("lr", 1e-4),
            weight_decay=t.get("weight_decay", 1e-4),
            device=device,
            out_dir=str(out_dir),
            debug_out_dir=str(out_dir),
            train_transform=None,
            val_transform=None,
            model_kwargs=m,
        )
    finally:
        finish_run()


def _run_full_train(args):
    cfg = build_config_from_args(args)
    t = cfg.get("trainer", {})
    m = cfg.get("model", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed, deterministic=args.deterministic)

    run = init_wandb_run(
        config=cfg.to_dict(),
        project="mask2former-forgery",
        job_type="full_train",
        group="full_train",
        name=t.get("name", Path(t.get("save_path", "model")).stem),
    )

    try:
        run_full_train(
            num_epochs=t.get("epochs", 30),
            batch_size=t.get("batch_size", 4),
            lr=t.get("lr", 1e-4),
            weight_decay=t.get("weight_decay", 1e-4),
            device=device,
            train_transform=None,
            save_path=t.get(
                "save_path",
                "weights/full_train/model_full_data_baseline.pth",
            ),
            model_kwargs=m,
        )
    finally:
        finish_run()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("Mask2Former Forgery CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_cv_subparser(subparsers)
    _add_full_train_subparser(subparsers)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "cv":
        _run_cv(args)
    elif args.command == "full-train":
        _run_full_train(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
