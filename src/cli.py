# src/cli.py
"""
Unified, simplified CLI for Mask2Former forgery experiments.

All configuration comes from YAML + CLI overrides.
CLI flags exist only for truly top-level concerns (command, device, seed).

Usage
-----
python -m src.cli cv -c base_v2.yaml -o trainer.lr=3e-4
python -m src.cli full-train -c best_oof.yaml
"""

import argparse
import inspect
from pathlib import Path

import torch

from src.training.train_cv import run_cv
from src.training.train_full import run_full_train
from src.utils.seed_logging_utils import setup_seed
from src.utils.wandb_utils import init_wandb_run, finish_run
from src.utils.config_utils import add_config_arguments, build_config_from_args, sanitize_model_kwargs


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _resolve_device(device_flag: str) -> torch.device:
    device_flag = (device_flag or "auto").lower()
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _filter_model_kwargs(model_kwargs: dict) -> dict:
    """
    Remove stale/unknown knobs so old YAMLs don't crash newer model ctors.
    """
    try:
        # Prefer v2 if present
        from src.models.mask2former_v2 import Mask2FormerForgeryModel  # noqa: WPS433
        sig = inspect.signature(Mask2FormerForgeryModel.__init__)
        allowed = {k for k in sig.parameters.keys() if k != "self"}
        return {k: v for k, v in model_kwargs.items() if k in allowed}
    except Exception:
        # If import/signature fails, pass through sanitized kwargs
        return dict(model_kwargs)


def _wandb_kwargs(cfg: dict, job_type: str, group: str, name: str) -> dict:
    w = cfg.get("wandb", {}) or {}
    return dict(
        config=cfg,
        project=w.get("project", "mask2former-forgery"),
        entity=w.get("entity", None),
        mode=w.get("mode", "online"),
        job_type=job_type,
        group=group,
        name=name,
    )


# ---------------------------------------------------------------------
# Subparsers
# ---------------------------------------------------------------------

def _add_cv_subparser(subparsers):
    p = subparsers.add_parser("cv", help="Run K-fold CV and produce OOF results")
    add_config_arguments(p)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    p.set_defaults(deterministic=True)


def _add_full_train_subparser(subparsers):
    p = subparsers.add_parser("full-train", help="Train on full dataset and save weights")
    add_config_arguments(p)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    p.set_defaults(deterministic=True)


# ---------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------

def _run_cv(args):
    cfg = build_config_from_args(args)
    cfg_dict = cfg.to_dict()
    t = cfg_dict.get("trainer", {}) or {}
    m = cfg_dict.get("model", {}) or {}

    device = _resolve_device(args.device)
    setup_seed(args.seed, deterministic=args.deterministic)

    # sanitize + drop stale knobs not in current ctor
    model_kwargs = _filter_model_kwargs(sanitize_model_kwargs(m))

    base_out_dir = Path(t.get("out_dir", "experiments/oof_results"))
    run_name = t.get("name", "cv")
    out_dir = base_out_dir / run_name

    init_wandb_run(**_wandb_kwargs(cfg_dict, job_type="cv", group="cv", name=run_name))

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
            model_kwargs=model_kwargs,
        )
    finally:
        finish_run()


def _run_full_train(args):
    cfg = build_config_from_args(args)
    cfg_dict = cfg.to_dict()
    t = cfg_dict.get("trainer", {}) or {}
    m = cfg_dict.get("model", {}) or {}

    device = _resolve_device(args.device)
    setup_seed(args.seed, deterministic=args.deterministic)

    # sanitize + drop stale knobs not in current ctor
    model_kwargs = _filter_model_kwargs(sanitize_model_kwargs(m))

    save_path = t.get("save_path", "weights/full_train/model_full_data_baseline.pth")
    run_name = t.get("name", Path(save_path).stem)

    init_wandb_run(**_wandb_kwargs(cfg_dict, job_type="full_train", group="full_train", name=run_name))

    try:
        run_full_train(
            num_epochs=t.get("epochs", 30),
            batch_size=t.get("batch_size", 4),
            lr=t.get("lr", 1e-4),
            weight_decay=t.get("weight_decay", 1e-4),
            device=device,
            train_transform=None,
            save_path=save_path,
            model_kwargs=model_kwargs,
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
