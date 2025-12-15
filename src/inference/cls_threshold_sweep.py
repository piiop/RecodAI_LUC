# src/inference/cls_threshold_sweep.py

"""
Sweep cls_threshold values for a fixed full-train model,
in the same style as auth_gate_sweep.py.

- Unified YAML config + CLI overrides
- Explicit inference overrides per run
- Auth gate disabled by default
- Per-threshold CSVs + summary.json
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataloader import ForgeryDataset, get_val_transform
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
    return ForgeryDataset(transform=transform)


def load_model(weights: str, model_cfg: dict, device: torch.device):
    mk = sanitize_model_kwargs(model_cfg)
    # pop the default to ensure we're using -1
    mk.pop("auth_gate_forged_threshold", None)
    model = Mask2FormerForgeryModel(
        **mk,
        auth_gate_forged_threshold=-1.0,  # disable gate by default
    ).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_cls_threshold(
    model,
    dataset,
    solution_df,
    cls_threshold: float,
    device,
    batch_size: int,
    mask_threshold: float,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    preds: List[str] = [None] * len(dataset)

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(
                images,
                inference_overrides={
                    "cls_threshold": float(cls_threshold),
                    "mask_threshold": float(mask_threshold),
                    "auth_gate_forged_threshold": -1.0,
                },
            )

            for out, t in zip(outputs, targets):
                idx = int(t["image_id"].item())
                sample = dataset.samples[idx]

                if out["masks"].numel() == 0:
                    preds[idx] = "authentic"
                    continue

                # original H, W
                if sample["mask_path"] and Path(sample["mask_path"]).exists():
                    m = np.load(sample["mask_path"])
                    h, w = m.shape[-2:]
                else:
                    with Image.open(sample["image_path"]) as im:
                        w, h = im.size

                union = (out["masks"] > 0).any(dim=0).float()
                union = union[None, None]
                union = (
                    F.interpolate(union, size=(h, w), mode="nearest")
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                preds[idx] = rle_encode(union)

    submission = pd.DataFrame(
        {"row_id": solution_df["row_id"], "annotation": preds}
    )

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
    p = argparse.ArgumentParser("CLS threshold sweep")
    add_config_arguments(p)

    p.add_argument("--weights", required=True)
    p.add_argument(
        "--cls_thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    p.add_argument("--out_dir", default="experiments/cls_threshold_sweep")
    p.add_argument("--seed", type=int, default=42)

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
    solution_df, _ = build_solution_df(dataset)

    model = load_model(args.weights, model_cfg, device)

    mask_threshold = model_cfg.get("default_mask_threshold", 0.5)

    results = []

    for ct in args.cls_thresholds:
        score, submission = run_cls_threshold(
            model=model,
            dataset=dataset,
            solution_df=solution_df,
            cls_threshold=ct,
            device=device,
            batch_size=trainer_cfg.get("batch_size", 4),
            mask_threshold=mask_threshold,
        )

        csv_path = out_dir / f"oof_cls_{ct:.3f}.csv"
        submission.to_csv(csv_path, index=False)

        results.append(
            {"cls_threshold": ct, "score": score, "csv": str(csv_path)}
        )

        print(f"cls_threshold={ct:.3f} score={score:.6f}")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("\nSorted results:")
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        print(f"cls_threshold={r['cls_threshold']:.3f} score={r['score']:.6f}")


if __name__ == "__main__":
    main()
