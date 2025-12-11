# src/cli.py

"""
Unified CLI entrypoint for local Kaggle forgery experiments.

Subcommands
-----------
1) cv
   - Run K-fold cross-validation on local training data.
   - Produces OOF predictions and metrics using the official Kaggle metric.
   - Uses src.training.train_cv.run_cv under the hood (no Kaggle submission).

2) full-train
   - Train on the full training set with chosen hyperparameters.
   - Saves model weights to disk for later use in a Kaggle submission notebook.
   - Uses src.training.train_full.run_full_train under the hood.

Example usage
-------------
# OOF CV run (local metric for experimentation/ablations)
python -m src.cli cv \
  --train_authentic /data/train_images/authentic \
  --train_forged   /data/train_images/forged \
  --train_masks    /data/train_masks \
  --n_folds 5 --epochs 3 --batch_size 4 --lr 1e-4

# Full-data training to produce weights for Kaggle
python -m src.cli full-train \
  --train_authentic /data/train_images/authentic \
  --train_forged   /data/train_images/forged \
  --train_masks    /data/train_masks \
  --epochs 30 --batch_size 4 --lr 1e-4 \
  --save_path weights/full_train/model_full_best_oof.pth

In the Kaggle submission notebook, you’ll manually upload the saved .pth
and load it into Mask2FormerForgeryModel before running test inference.
"""

import argparse
from pathlib import Path

import torch

from src.training.train_cv import run_cv
from src.training.train_full import run_full_train
from src.utils.seed_logging_utils import setup_seed, log_seed_info
from src.utils.wandb_utils import (
    init_wandb_run,
    log_config,
    finish_run,
)


# ---------------------------------------------------------------------------
# Subcommand: CV
# ---------------------------------------------------------------------------


def _add_cv_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "cv",
        help="Run K-fold CV + OOF metric on local training data.",
    )

    # Data paths
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

    # CV / training hyperparams (mirrors src/training/train_cv.py)
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs per fold (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="experiments/oof_results",
        help="Directory to store OOF predictions/metrics",
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


def _run_cv_from_args(args: argparse.Namespace) -> None:
    paths = {
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

    seed_info = setup_seed(args.seed, deterministic=args.deterministic)
    log_seed_info(seed_info)

    # wandb run for this CV experiment (optional; no-op if wandb disabled)
    run = init_wandb_run(
        config=vars(args),
        project="mask2former-forgery",
        job_type="cv",
        group="cv",
        name=f"cv_{Path(args.out_dir).name}",
    )
    log_config(vars(args))

    try:
        run_cv(
            paths=paths,
            num_folds=args.n_folds,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            train_transform=None,  # plug in your train/val transforms when ready
            val_transform=None,
            out_dir=args.out_dir,
        )
    finally:
        finish_run()


# ---------------------------------------------------------------------------
# Subcommand: full-train
# ---------------------------------------------------------------------------


def _add_full_train_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "full-train",
        help="Train on the full dataset and save weights for Kaggle submission.",
    )

    # Data paths
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

    # Training hyperparams (mirrors src/training/train_full.py)
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs (default: 30, typically best from CV)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="weights/full_train/model_full_data_baseline.pth",
        help=(
            "Where to save model weights "
            "(default: weights/full_train/model_full_data_baseline.pth)"
        ),
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


def _run_full_train_from_args(args: argparse.Namespace) -> None:
    paths = {
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

    seed_info = setup_seed(args.seed, deterministic=args.deterministic)
    log_seed_info(seed_info)

    # wandb run for full-data training (optional; no-op if wandb disabled)
    run = init_wandb_run(
        config=vars(args),
        project="mask2former-forgery",
        job_type="full_train",
        group="full_train",
        name=f"full_{Path(args.save_path).stem}",
    )
    log_config(vars(args))

    try:
        run_full_train(
            paths=paths,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            train_transform=None,  # plug in same train transform used in best CV
            save_path=args.save_path,
        )
    finally:
        finish_run()


# ---------------------------------------------------------------------------
# Top-level parser / main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified CLI for local Kaggle forgery experiments:\n"
            "  - 'cv' for OOF cross-validation\n"
            "  - 'full-train' for full-data training & saving weights"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_cv_subparser(subparsers)
    _add_full_train_subparser(subparsers)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "cv":
        _run_cv_from_args(args)
    elif args.command == "full-train":
        _run_full_train_from_args(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
