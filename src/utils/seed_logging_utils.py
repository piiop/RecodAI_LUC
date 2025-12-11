"""
src/utils/seed_logging_utils.py

Helpers for:
  - Setting all relevant random seeds (Python, NumPy, PyTorch, CUDA)
  - Making PyTorch runs deterministic (optional)
  - Collecting lightweight run metadata for logging / wandb

Typical usage:

    from src.utils.seed_logging_utils import setup_seed, log_seed_info

    seed_info = setup_seed(42, deterministic=True)
    log_seed_info(seed_info)        # prints a one-line summary
    # wandb.log(seed_info.to_flat_dict(prefix="meta/"))  # if desired
"""

from __future__ import annotations

import os
import sys
import random
import socket
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, TextIO
import wandb
import numpy as np
import torch


@dataclass
class SeedInfo:
    """Container for seed + environment metadata."""

    seed: int
    deterministic: bool

    # PyTorch / CUDA
    torch_version: str = field(default_factory=lambda: torch.__version__)
    cuda_available: bool = field(default_factory=torch.cuda.is_available)
    cuda_device_count: int = field(default_factory=torch.cuda.device_count)
    cudnn_available: bool = field(default_factory=lambda: torch.backends.cudnn.is_available())
    cudnn_deterministic: bool = field(
        default_factory=lambda: torch.backends.cudnn.deterministic
    )
    cudnn_benchmark: bool = field(default_factory=lambda: torch.backends.cudnn.benchmark)

    # NumPy / Python
    numpy_version: str = field(default_factory=lambda: np.__version__)
    python_version: str = field(default_factory=lambda: sys.version.replace("\n", " "))
    python_hash_seed: Optional[int] = None

    # System / provenance
    hostname: str = field(default_factory=socket.gethostname)
    command: str = field(default_factory=lambda: " ".join(sys.argv))
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a nested dict representation (safe for JSON logging)."""
        return asdict(self)

    def to_flat_dict(self, prefix: str = "meta/") -> Dict[str, Any]:
        """
        Flatten into a 1-level dict, suitable for wandb or tensorboard.

        Example key: 'meta/seed', 'meta/cuda_available', ...
        """
        base = asdict(self)
        return {f"{prefix}{k}": v for k, v in base.items()}


def _get_python_hash_seed() -> Optional[int]:
    val = os.environ.get("PYTHONHASHSEED")
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _try_get_git_commit() -> Optional[str]:
    """Return short git commit hash if available, otherwise None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _set_torch_determinism(deterministic: bool) -> None:
    """
    Configure PyTorch's deterministic settings in a version-tolerant way.
    """
    # Ensure cuBLAS uses deterministic workspace when requested
    if deterministic:
        # Only set if not already provided by user / shell
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # cuDNN flags
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # use_deterministic_algorithms is available in newer PyTorch
    try:
        # warn_only=True avoids hard errors for some ops
        torch.use_deterministic_algorithms(deterministic, warn_only=True)  # type: ignore[arg-type]
    except TypeError:
        # Older signature without warn_only
        try:
            torch.use_deterministic_algorithms(deterministic)  # type: ignore[call-arg]
        except Exception:
            pass
    except AttributeError:
        # Very old PyTorch; nothing more to do
        pass



def setup_seed(seed: int, deterministic: bool = True) -> SeedInfo:
    """
    Set all relevant RNG seeds for Python, NumPy and PyTorch, and optionally
    enable deterministic behavior for PyTorch/cuDNN.

    Args:
        seed: Base integer seed.
        deterministic: If True, configures PyTorch/cuDNN for deterministic
                       behavior (may have some performance cost).

    Returns:
        SeedInfo with metadata you can log to stdout / wandb / JSON.
    """
    # Python hash seed (affects object hashing / dict iteration in some cases)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python / NumPy RNGs
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _set_torch_determinism(deterministic)

    info = SeedInfo(
        seed=seed,
        deterministic=deterministic,
        python_hash_seed=_get_python_hash_seed(),
        git_commit=_try_get_git_commit(),
    )
    return info


def log_seed_info(
    info: SeedInfo,
    fp: TextIO = sys.stdout,
    prefix: str = "[seed]",
) -> None:
    """
    Print a compact one-line summary of seed + device metadata.

    Args:
        info: SeedInfo returned from setup_seed.
        fp:   Where to write (default: stdout).
        prefix: String prefix for the log line.
    """
    msg_parts = [
        f"seed={info.seed}",
        f"deterministic={info.deterministic}",
        f"cuda={info.cuda_available} (n={info.cuda_device_count})",
        f"cudnn_det={info.cudnn_deterministic}",
        f"cudnn_bench={info.cudnn_benchmark}",
        f"torch={info.torch_version}",
        f"numpy={info.numpy_version}",
        f"host={info.hostname}",
    ]
    if info.git_commit is not None:
        msg_parts.append(f"git={info.git_commit}")

    print(f"{prefix} " + " | ".join(msg_parts), file=fp)


def log_seed_to_wandb(info: SeedInfo, run: Any = None, prefix: str = "meta/") -> None:
    """
    Convenience helper to push seed metadata into an existing wandb run.

    This is intentionally optional to keep this module usable without wandb.

    Args:
        info: SeedInfo from setup_seed.
        run:  Existing wandb.Run instance (if None, uses wandb.run).
        prefix: Key prefix for flattened metadata.
    """

    if run is None:
        run = wandb.run

    if run is None:
        return

    flat = info.to_flat_dict(prefix=prefix)

    # Put 'seed' and 'deterministic' into config for visibility
    run.config.update(
        {
            "seed": info.seed,
            "deterministic": info.deterministic,
        },
        allow_val_change=True,
    )
    # Log the rest as a single step-0 record
    run.log(flat, step=0)
