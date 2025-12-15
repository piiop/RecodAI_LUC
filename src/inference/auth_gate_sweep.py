# src/inference/auth_gate_sweep.py

"""
Sweep authenticity gate thresholds for a fixed full-train model,
using the unified YAML config system.

Key fixes vs old version:
- Use ForgeryDataset(..., is_train=False) for a stable eval split
- Use shared detection_collate_fn (consistent with other inference scripts)
- Make mask/cls thresholds configurable (and applied via inference_overrides)
- Robustly handle weights that may be stored as {"state_dict": ...} or raw state_dict
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataloader import (
    ForgeryDataset,
    get_val_transform,
    detection_collate_fn,
)
from src.models.mask2former_v1 import Mask2FormerForgeryModel
from src.models.kaggle_metric import score as kaggle_score
from src.inference.postprocess import rle_encode
from src.training.train_cv import build_solution_df
from src.utils.seed_logging_utils import setup_seed
from src.utils.config_utils import (
    add_config_arguments,
    build_config_from_args,
    sanitize_model_kwargs,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def build_dataset(img_size: int) -> ForgeryDataset:
    transform = get_val_transform(img_size=img_size)
    return ForgeryDataset(transform=transform, is_train=False)


def _load_state_dict(weights_path: str, device: torch.device) -> Dict[str, Any]:
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict):
        # common wrappers
        for k in ("state_dict", "model_state_dict", "model"):
            if k in state and isinstance(state[k], dict):
                return state[k]
    return state


def load_model(weights: str, model_cfg: dict, device: torch.device) -> Mask2FormerForgeryModel:
    mk = sanitize_model_kwargs(model_cfg)

    # pop the default to ensure we're using -1
    mk.pop("auth_gate_forged_threshold", None)

    # Start with gate disabled so only the sweep value matters.
    model = Mask2FormerForgeryModel(
        **mk,
        auth_gate_forged_threshold=-1.0,
    ).to(device)

    sd = _load_state_dict(weights, device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _get_original_hw(sample: dict) -> Tuple[int, int]:
    """
    Returns (h, w) for the *original* image/mask.
    """
    mp = sample.get("mask_path", None)
    if mp and Path(mp).exists():
        m = np.load(mp)
        # handle [N,H,W] or [H,W]
        if m.ndim == 3:
            h, w = int(m.shape[1]), int(m.shape[2])
        else:
            h, w = int(m.shape[0]), int(m.shape[1])
        return h, w

    with Image.open(sample["image_path"]) as im:
        w, h = im.size
    return int(h), int(w)


def run_gate(
    model: Mask2FormerForgeryModel,
    dataset: ForgeryDataset,
    solution_df: pd.DataFrame,
    gate: float,
    device: torch.device,
    batch_size: int,
    *,
    mask_threshold: float,
    cls_threshold: float,
    num_workers: int = 4,
) -> Tuple[float, pd.DataFrame]:
    model.auth_gate_forged_threshold = float(gate)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=detection_collate_fn,
    )

    preds: List[str] = [None] * len(dataset)

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]

            # Ensure we apply the exact thresholds we want (not whatever was trained into the model defaults).
            outputs = model(
                images,
                targets=None,
                inference_overrides={
                    "mask_threshold": float(mask_threshold),
                    "cls_threshold": float(cls_threshold),
                    # IMPORTANT: do NOT override auth_gate here; we sweep via model.auth_gate_forged_threshold
                },
            )

            for out, t in zip(outputs, targets):
                idx = int(t["image_id"].item())
                sample = dataset.samples[idx]

                if out["masks"].numel() == 0:
                    preds[idx] = "authentic"
                    continue

                h, w = _get_original_hw(sample)

                # union all predicted masks then upsample to original size
                union = (out["masks"] > 0).any(dim=0).float()  # [Hm, Wm]
                union = union[None, None]  # [1,1,Hm,Wm]
                union = (
                    F.interpolate(union, size=(h, w), mode="nearest")
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                preds[idx] = rle_encode(union)

    submission = pd.DataFrame({"row_id": solution_df["row_id"], "annotation": preds})

    score = kaggle_score(
        solution_df.copy(),
        submission.copy(),
        row_id_column_name="row_id",
    )
    return float(score), submission


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("Auth gate sweep")
    add_config_arguments(p)

    p.add_argument("--weights", required=True)
    p.add_argument("--gates", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--out_dir", default="experiments/auth_gate_sweep")
    p.add_argument("--seed", type=int, default=42)

    # Optional explicit thresholds (default to config/model defaults)
    p.add_argument("--mask_threshold", type=float, default=None)
    p.add_argument("--cls_threshold", type=float, default=None)

    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = build_config_from_args(args)

    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg.get("model", {})
    trainer_cfg = cfg.get("trainer", {})
    data_cfg = cfg.get("data", {})

    dataset = build_dataset(img_size=data_cfg.get("img_size", 256))

    # Build solution_df against the same underlying samples list.
    solution_df, _ = build_solution_df(dataset)

    model = load_model(args.weights, model_cfg, device)

    # Choose thresholds: CLI > config > model defaults
    mask_thr = (
        float(args.mask_threshold)
        if args.mask_threshold is not None
        else float(model_cfg.get("default_mask_threshold", getattr(model, "default_mask_threshold", 0.5)))
    )
    cls_thr = (
        float(args.cls_threshold)
        if args.cls_threshold is not None
        else float(model_cfg.get("default_cls_threshold", getattr(model, "default_cls_threshold", 0.5)))
    )

    results = []
    for gate in args.gates:
        score, submission = run_gate(
            model=model,
            dataset=dataset,
            solution_df=solution_df,
            gate=gate,
            device=device,
            batch_size=int(trainer_cfg.get("batch_size", 4)),
            mask_threshold=mask_thr,
            cls_threshold=cls_thr,
            num_workers=int(args.num_workers),
        )

        csv_path = out_dir / f"oof_gate_{gate:.3f}.csv"
        submission.to_csv(csv_path, index=False)

        results.append(
            {
                "gate": float(gate),
                "score": float(score),
                "csv": str(csv_path),
                "mask_threshold": float(mask_thr),
                "cls_threshold": float(cls_thr),
            }
        )

        print(f"gate={gate:.3f} score={score:.6f}")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("\nSorted results:")
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        print(f"gate={r['gate']:.3f} score={r['score']:.6f}")


if __name__ == "__main__":
    main()
