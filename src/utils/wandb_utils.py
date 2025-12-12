# src/utils/wandb_utils.py

"""
Lightweight helpers for integrating Weights & Biases (wandb) into the
training scripts (CV + full-train).

Features:
- Safe initialization (no-op if wandb isn't installed or disabled)
- Config logging from dict / argparse.Namespace / simple objects
- Metric logging with optional prefixes (e.g. "train/", "val/", "fold_0/")
- Simple artifact logging for model checkpoints and OOF outputs
"""

from __future__ import annotations

import os
import argparse
from typing import Any, Dict, Mapping, Optional, Sequence, Union

try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - purely defensive
    wandb = None
    _WANDB_AVAILABLE = False

ConfigLike = Union[Mapping[str, Any], "argparse.Namespace", Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_config_dict(config: Optional[ConfigLike]) -> Dict[str, Any]:
    """
    Convert various config-like objects to a plain dict.

    Supported:
        - dict-like mappings
        - argparse.Namespace
        - objects with __dict__
    """
    if config is None:
        return {}

    # Mapping (dict, etc.)
    if isinstance(config, Mapping):
        return dict(config)

    # argparse.Namespace or similar
    if hasattr(config, "__dict__"):
        return {
            k: v
            for k, v in vars(config).items()
            if not k.startswith("_")
        }

    # Fallback: try to use __dict__ directly
    try:
        return dict(config.__dict__)
    except Exception:
        # Last-resort: nothing to do, but avoid crashing
        return {}


def _wandb_enabled() -> bool:
    """Return True if wandb is importable and not globally disabled."""
    if not _WANDB_AVAILABLE:
        return False

    # Allow global disabling via env var
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_wandb_run(
    config: Optional[ConfigLike] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    mode: Optional[str] = None,
    job_type: Optional[str] = None,
    group: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    notes: Optional[str] = None,
    run_id: Optional[str] = None,
    resume: Optional[Union[bool, str]] = None,
):
    """
    Initialize a wandb run (or return None if wandb is unavailable / disabled).

    Typical use in train scripts:
        run = init_wandb_run(
            config=vars(args),
            project="forgery-mask2former",
            job_type="cv" or "full_train",
            group="exp_001",
            name="cv_fold_0" or "full_train_best_oof",
            tags=["mask2former", "kaggle"],
        )

    Args mirror `wandb.init` but are kept minimal here.
    """
    if not _wandb_enabled():
        print("[wandb_utils] wandb is not available or disabled; running without logging.")
        return None

    cfg_dict = _to_config_dict(config)

    # Let env vars override project/entity/mode if set.
    init_kwargs: Dict[str, Any] = {
        "project": project,
        "entity": entity,
        "config": cfg_dict if cfg_dict else None,
        "job_type": job_type,
        "group": group,
        "name": name,
        "tags": list(tags) if tags is not None else None,
        "notes": notes,
        "id": run_id,
        "resume": resume,
    }

    # Drop keys that are None to keep wandb.init clean
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    if mode is not None:
        init_kwargs["mode"] = mode  # e.g. "online", "offline", "disabled"

    run = wandb.init(**init_kwargs)
    return run


def log_config(config: Optional[ConfigLike]) -> None:
    """
    Update the current wandb run's config with the provided values.

    Safe no-op if wandb run isn't active.
    """
    if not _wandb_enabled() or wandb.run is None:
        return

    cfg_dict = _to_config_dict(config)
    if not cfg_dict:
        return

    wandb.config.update(cfg_dict, allow_val_change=True)


def log_metrics(
    metrics: Mapping[str, Union[int, float]],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
    commit: bool = True,
) -> None:
    """
    Log scalar metrics to the current wandb run.

    Args:
        metrics: dict of metric_name -> value
        step: optional global step
        prefix: if provided, prepend e.g. "train/" or "val/" to keys
        commit: passed through to wandb.log
    """
    if not _wandb_enabled() or wandb.run is None:
        return

    if prefix:
        prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}
    else:
        prefixed = dict(metrics)

    if step is not None:
        prefixed["step"] = step

    wandb.log(prefixed, commit=commit)


def log_epoch_metrics(
    stage: str,
    metrics: Mapping[str, Union[int, float]],
    epoch: int,
    global_step: Optional[int] = None,
) -> None:
    """
    Convenience wrapper for logging per-epoch metrics.

    Example:
        log_epoch_metrics("train", {"loss": 0.123}, epoch=3)
        -> logs "train/loss" and "epoch"
    """
    merged = dict(metrics)
    merged["epoch"] = epoch
    prefix = f"{stage}/" if stage else None
    log_metrics(merged, step=global_step, prefix=prefix)


def set_summary(
    key: str,
    value: Union[int, float, str, Dict[str, Any]],
) -> None:
    """
    Set a summary field on the current run (e.g. final CV score, OOF score).

    Safe no-op if wandb run isn't active.
    """
    if not _wandb_enabled() or wandb.run is None:
        return

    wandb.run.summary[key] = value


def log_best_metric(
    metric_name: str,
    value: Union[int, float],
    higher_is_better: bool = True,
    summary_key: Optional[str] = None,
) -> None:
    """
    Keep track of the best value of a metric in wandb.run.summary.

    Example:
        log_best_metric("mean_cv", 0.835, higher_is_better=True)
    """
    if not _wandb_enabled() or wandb.run is None:
        return

    key = summary_key or f"best_{metric_name}"
    summary = wandb.run.summary

    # Avoid `key in summary` because wandb.Summary's membership
    # check can hit __getitem__ with integer indices and raise KeyError.
    try:
        prev = summary[key]
    except KeyError:
        # First time we see this metric: just set it.
        summary[key] = value
        return
    except Exception:
        # If summary behaves unexpectedly, fall back to simply setting it.
        summary[key] = value
        return

    try:
        is_better = (value > prev) if higher_is_better else (value < prev)
    except Exception:
        is_better = False

    if is_better:
        summary[key] = value



def log_artifact(
    path: str,
    name: Optional[str] = None,
    type: str = "model",
    aliases: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a file (e.g. model checkpoint, OOF CSV) as a wandb artifact.

    Typical usage:
        log_artifact(
            path="weights/full_train/model_full_data_baseline.pth",
            name="model_full_data_baseline",
            type="model",
            aliases=["full_train", "latest"],
        )
    """
    if not _wandb_enabled() or wandb.run is None:
        return

    file_path = os.path.abspath(path)
    if not os.path.exists(file_path):
        print(f"[wandb_utils] Artifact path does not exist: {file_path}")
        return

    art_name = name or os.path.basename(file_path)
    artifact = wandb.Artifact(
        name=art_name,
        type=type,
        metadata=metadata or {},
    )
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)


def finish_run() -> None:
    """
    Explicitly finish the current wandb run.

    Usually optional, but nice to call at the end of main().
    """
    if not _wandb_enabled() or wandb.run is None:
        return
    wandb.finish()
