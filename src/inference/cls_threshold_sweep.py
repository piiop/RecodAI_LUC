# src/inference/cls_threshold_sweep.py

"""
Sweep inference thresholds for a fixed full-train model (v2 architecture).

Aligned with Mask2FormerForgeryModel.inference() logic:
  qscore[q] = sigmoid(class_logit[q]) * mean(sigmoid(mask_logit[q]))
  presence_prob = max_q qscore[q]

Per image:
  1) compute qscore + presence_prob
  2) select queries via:
        - topk by qscore (if topk provided and >0), else
        - qscore_threshold (if provided), else
        - keep all
     then apply optional min_mask_mass filter
     then optional presence_threshold gate (if fails => predict authentic)
  3) threshold kept masks by mask_threshold
  4) union kept masks, upsample to original H,W
  5) optional area_threshold: if union_area_ratio < area_threshold => predict "authentic"
  6) else encode union as RLE

Scores with official Kaggle metric, writes per-config submission CSV + summary.json.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataloader import ForgeryDataset, get_val_transform
from src.models.mask2former_v2 import Mask2FormerForgeryModel
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
    if "auth_penalty_cls_threshold" in mk:
        print("FOUND legacy key in mk:", mk["auth_penalty_cls_threshold"])
    # Drop any stale inference knobs from older configs; we control them in the sweep.
    mk.pop("auth_gate_forged_threshold", None)
    mk.pop("default_cls_threshold", None)
    mk.pop("default_mask_threshold", None)
    mk.pop("default_qscore_threshold", None)
    mk.pop("default_topk", None)
    mk.pop("default_min_mask_mass", None)
    mk.pop("default_presence_threshold", None)

    model = Mask2FormerForgeryModel(
        **mk,
        default_mask_threshold=0.0,
        default_qscore_threshold=None,
        default_topk=None,
        default_min_mask_mass=None,
        default_presence_threshold=None,
    ).to(device)

    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _format_tag(x: float) -> str:
    return f"{x:.6g}".replace(".", "p")


@torch.no_grad()
def run_sweep_config(
    model: Mask2FormerForgeryModel,
    infer_dataset: ForgeryDataset,
    solution_df: pd.DataFrame,
    hws: List[Tuple[int, int]],
    *,
    mask_threshold: float,
    qscore_threshold: Optional[float],
    topk: Optional[int],
    min_mask_mass: Optional[float],
    presence_threshold: Optional[float],
    area_threshold: Optional[float],
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

        # v2 forward_logits returns (mask_logits, class_logits)
        mask_logits, class_logits = model.forward_logits(images)

        mask_probs = torch.sigmoid(mask_logits)          # [B,Q,Hm,Wm]
        cls_probs = torch.sigmoid(class_logits)          # [B,Q]
        mask_mass = mask_probs.flatten(2).mean(-1)       # [B,Q]
        qscore = cls_probs * mask_mass                   # [B,Q]
        presence_prob = qscore.max(dim=1).values         # [B]

        B, Q = cls_probs.shape

        # indices in global dataset order
        idxs = [int(t["image_id"].item()) for t in targets]

        for bi, idx in enumerate(idxs):
            # ----- selection -----
            keep = torch.zeros((Q,), dtype=torch.bool, device=device)

            if topk is not None and int(topk) > 0:
                k = min(int(topk), Q)
                top_idx = torch.topk(qscore[bi], k=k, largest=True).indices
                keep[top_idx] = True
            elif qscore_threshold is not None:
                keep = qscore[bi] > float(qscore_threshold)
            else:
                keep[:] = True

            if min_mask_mass is not None:
                keep = keep & (mask_mass[bi] > float(min_mask_mass))

            if presence_threshold is not None:
                if not bool((presence_prob[bi] > float(presence_threshold)).item()):
                    preds[idx] = "authentic"
                    continue

            if int(keep.sum().item()) == 0:
                preds[idx] = "authentic"
                continue

            kept_probs = mask_probs[bi, keep]  # [K,Hm,Wm]
            if kept_probs.numel() == 0:
                preds[idx] = "authentic"
                continue

            masks_bool = kept_probs > float(mask_threshold)
            if (not bool(masks_bool.any().item())):
                preds[idx] = "authentic"
                continue

            union_small = masks_bool.any(dim=0).float()[None, None]  # [1,1,Hm,Wm]
            h, w = hws[idx]
            union_up = (
                F.interpolate(union_small, size=(h, w), mode="nearest")
                .squeeze(0)
                .squeeze(0)
            )  # [H,W] float

            if area_threshold is not None:
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
    p = argparse.ArgumentParser("qscore × topk/min_mass/presence × area sweep")
    add_config_arguments(p)

    p.add_argument("--weights", required=True)

    # Sweep knobs
    p.add_argument("--qscore_thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05, 0.08])
    p.add_argument("--topk", type=int, nargs="+", default=[0, 1, 2, 3, 5],
                   help="0 means disabled (use qscore_threshold instead).")
    p.add_argument("--min_mask_mass", type=float, nargs="+", default=[0.0, 0.001, 0.002, 0.005])

    p.add_argument("--presence_thresholds", type=float, nargs="+", default=[-1.0, 0.01, 0.02, 0.03],
                   help="-1 means disabled (no gate).")

    p.add_argument("--area_thresholds", type=float, nargs="+", default=[-1.0, 0.0005, 0.001, 0.002, 0.005],
                   help="-1 means disabled.")

    p.add_argument("--mask_threshold", type=float, default=None,
                   help="If not set, uses model_cfg.default_mask_threshold (or 0.5 fallback).")

    p.add_argument("--out_dir", default="experiments/qscore_sweep")
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
    solution_df = build_solution_df(full_dataset)
    hws = precompute_hw(full_dataset)


    model = load_model(args.weights, model_cfg, device)

    if args.mask_threshold is not None:
        mask_threshold = float(args.mask_threshold)
    else:
        mask_threshold = float(model_cfg.get("default_mask_threshold", 0.5))

    results: List[Dict] = []

    for qt in args.qscore_thresholds:
        for tk in args.topk:
            for mm in args.min_mask_mass:
                for pt in args.presence_thresholds:
                    for at in args.area_thresholds:
                        qscore_thr = None if tk and int(tk) > 0 else float(qt)
                        topk = None if int(tk) <= 0 else int(tk)

                        pres_thr = None if float(pt) < 0 else float(pt)
                        area_thr = None if float(at) < 0 else float(at)
                        min_mass = None if float(mm) <= 0 else float(mm)

                        score, submission = run_sweep_config(
                            model=model,
                            infer_dataset=infer_dataset,
                            solution_df=solution_df,
                            hws=hws,
                            mask_threshold=mask_threshold,
                            qscore_threshold=qscore_thr,
                            topk=topk,
                            min_mask_mass=min_mass,
                            presence_threshold=pres_thr,
                            area_threshold=area_thr,
                            device=device,
                            batch_size=batch_size,
                        )

                        tag = (
                            f"mt{_format_tag(mask_threshold)}_"
                            f"qt{_format_tag(qt)}_k{tk}_"
                            f"mm{_format_tag(mm)}_"
                            f"pt{_format_tag(pt)}_"
                            f"at{_format_tag(at)}"
                        )
                        csv_path = out_dir / f"oof_{tag}.csv"
                        submission.to_csv(csv_path, index=False)

                        row = {
                            "mask_threshold": float(mask_threshold),
                            "qscore_threshold": None if qscore_thr is None else float(qscore_thr),
                            "topk": None if topk is None else int(topk),
                            "min_mask_mass": None if min_mass is None else float(min_mass),
                            "presence_threshold": None if pres_thr is None else float(pres_thr),
                            "area_threshold": None if area_thr is None else float(area_thr),
                            "score": float(score),
                            "csv": str(csv_path),
                        }
                        results.append(row)

                        print(f"{tag} score={score:.6f}")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\nTop results:")
    for r in sorted(results, key=lambda x: x["score"], reverse=True)[:20]:
        print(
            f"score={r['score']:.6f} "
            f"mt={r['mask_threshold']:.4f} "
            f"qt={r['qscore_threshold']} "
            f"k={r['topk']} "
            f"mm={r['min_mask_mass']} "
            f"pt={r['presence_threshold']} "
            f"at={r['area_threshold']} "
            f"-> {r['csv']}"
        )


if __name__ == "__main__":
    main()
