# src/inference/cls_threshold_sweep.py

"""
Sweep (cls_threshold × area_threshold × topk) for a fixed full-train model.

- Uses forward_logits() to get raw per-query logits (so top-k is meaningful)
- Applies:
    1) pick top-k queries by cls_prob
    2) filter by cls_threshold
    3) threshold masks by mask_threshold
    4) union masks, upsample to original H,W
    5) if union_area_ratio < area_threshold -> predict "authentic"
- Scores with official Kaggle metric
- Writes per-config submission CSV + summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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

def collate_batch(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_datasets(img_size: int) -> Tuple[ForgeryDataset, ForgeryDataset]:
    """
    full_dataset: transform=None (for solution_df + original sizes)
    infer_dataset: resized/normalized (for model input)
    Ordering is identical, so indices align.
    """
    full_dataset = ForgeryDataset(transform=None)
    infer_dataset = ForgeryDataset(transform=get_val_transform(img_size=img_size))
    return full_dataset, infer_dataset


def precompute_hw(full_dataset: ForgeryDataset) -> List[Tuple[int, int]]:
    """List of (h,w) for each sample in dataset order."""
    hws: List[Tuple[int, int]] = []
    for s in full_dataset.samples:
        with Image.open(s["image_path"]) as im:
            w, h = im.size
        hws.append((h, w))
    return hws


def load_model(weights: str, model_cfg: dict, device: torch.device):
    mk = sanitize_model_kwargs(model_cfg)

    # Ensure inference gate is disabled (even if config has it)
        # Remove inference thresholds so we fully control them here
    mk.pop("auth_gate_forged_threshold", None)
    mk.pop("default_cls_threshold", None)
    mk.pop("default_mask_threshold", None)

    model = Mask2FormerForgeryModel(
        **mk,
        auth_gate_forged_threshold=-1.0,
        default_cls_threshold=0.0,   # we apply our own filtering
        default_mask_threshold=0.0,  # we apply our own filtering
    ).to(device)

    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _format_tag(x: float) -> str:
    # filename-safe compact float
    return f"{x:.6g}".replace(".", "p")


@torch.no_grad()
def run_sweep_config(
    model: Mask2FormerForgeryModel,
    infer_dataset: ForgeryDataset,
    solution_df: pd.DataFrame,
    hws: List[Tuple[int, int]],
    *,
    cls_threshold: float,
    area_threshold: float,
    topk: int,
    mask_threshold: float,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, pd.DataFrame]:
    loader = DataLoader(
        infer_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    preds: List[str] = ["authentic"] * len(infer_dataset)

    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]

        # Raw logits (no internal filtering)
        mask_logits, class_logits, _img_logits = model.forward_logits(images)

        # probs
        cls_probs = torch.sigmoid(class_logits)  # [B,Q]
        mask_probs = torch.sigmoid(mask_logits)  # [B,Q,H,W]

        B, Q = cls_probs.shape

        # indices in global dataset order
        idxs = [int(t["image_id"].item()) for t in targets]

        # per image
        for bi, idx in enumerate(idxs):
            # pick top-k by cls prob
            k = int(min(max(topk, 1), Q))
            top_idx = torch.topk(cls_probs[bi], k=k, largest=True).indices  # [k]

            # apply cls threshold
            keep = cls_probs[bi, top_idx] >= float(cls_threshold)
            if keep.sum().item() == 0:
                preds[idx] = "authentic"
                continue

            kept_idx = top_idx[keep]  # [K]

            # masks -> binary at mask_threshold
            masks_bool = mask_probs[bi, kept_idx] >= float(mask_threshold)  # [K,Hm,Wm]
            if masks_bool.numel() == 0 or (not bool(masks_bool.any().item())):
                preds[idx] = "authentic"
                continue

            union_small = masks_bool.any(dim=0).float()[None, None]  # [1,1,Hm,Wm]

            h, w = hws[idx]
            union_up = (
                F.interpolate(union_small, size=(h, w), mode="nearest")
                .squeeze(0)
                .squeeze(0)
            )  # [H,W] float

            area_ratio = float(union_up.sum().item() / max(h * w, 1))
            if area_ratio < float(area_threshold):
                preds[idx] = "authentic"
                continue

            union_np = union_up.cpu().numpy().astype(np.uint8)
            preds[idx] = rle_encode(union_np)

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
    p = argparse.ArgumentParser("CLS × area × topk sweep")
    add_config_arguments(p)

    p.add_argument("--weights", required=True)

    p.add_argument("--cls_thresholds", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.30])
    p.add_argument("--area_thresholds", type=float, nargs="+",
                   default=[0.0005, 0.001, 0.002, 0.005, 0.01])
    p.add_argument("--topk", type=int, nargs="+",
                   default=[1, 2, 3, 5])

    p.add_argument("--mask_threshold", type=float, default=None,
                   help="If not set, uses model_cfg.default_mask_threshold (or 0.5 fallback).")

    p.add_argument("--out_dir", default="experiments/cls_area_topk_sweep")
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

    img_size = int(data_cfg.get("img_size", 256))
    batch_size = int(trainer_cfg.get("batch_size", 4))

    full_dataset, infer_dataset = build_datasets(img_size=img_size)
    solution_df = build_solution_df(full_dataset)  # NOTE: returns only df (up-to-date)
    hws = precompute_hw(full_dataset)

    model = load_model(args.weights, model_cfg, device)

    if args.mask_threshold is not None:
        mask_threshold = float(args.mask_threshold)
    else:
        # keep backward compatibility with older configs; default to 0.5 if absent
        mask_threshold = float(model_cfg.get("default_mask_threshold", 0.5))

    results: List[Dict] = []

    for ct in args.cls_thresholds:
        for at in args.area_thresholds:
            for tk in args.topk:
                score, submission = run_sweep_config(
                    model=model,
                    infer_dataset=infer_dataset,
                    solution_df=solution_df,
                    hws=hws,
                    cls_threshold=float(ct),
                    area_threshold=float(at),
                    topk=int(tk),
                    mask_threshold=mask_threshold,
                    device=device,
                    batch_size=batch_size,
                )

                tag = f"ct{_format_tag(ct)}_at{_format_tag(at)}_k{tk}"
                csv_path = out_dir / f"oof_{tag}.csv"
                submission.to_csv(csv_path, index=False)

                row = {
                    "cls_threshold": float(ct),
                    "area_threshold": float(at),
                    "topk": int(tk),
                    "mask_threshold": float(mask_threshold),
                    "score": float(score),
                    "csv": str(csv_path),
                }
                results.append(row)

                print(f"{tag} score={score:.6f}")

    # write summary
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\nTop results:")
    for r in sorted(results, key=lambda x: x["score"], reverse=True)[:20]:
        print(
            f"ct={r['cls_threshold']:.4f} at={r['area_threshold']:.6g} "
            f"k={r['topk']} score={r['score']:.6f} -> {r['csv']}"
        )


if __name__ == "__main__":
    main()
