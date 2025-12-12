# src/inference/auth_gate_sweep.py

"""
Sweep authenticity gate thresholds for a *fixed* full-train model.

- Loads Mask2FormerForgeryModel + full-train weights.
- Runs inference on the *training set* for several gate values.
- Computes the official Kaggle metric for each gate using kaggle_metric.score.
- Optionally saves per-gate prediction CSVs and a summary JSON.

Usage example
-------------

python -m src.inference.auth_gate_sweep \
  --weights        weights/full_train/model_full_data_baseline.pth \
  --gates 0.3 0.4 0.5 0.6 0.7 \
  --batch_size 4 \
  --out_dir experiments/auth_gate_sweep
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataloader import ForgeryDataset, get_val_transform  # :contentReference[oaicite:0]{index=0}
from src.models.mask2former_v1 import Mask2FormerForgeryModel       # :contentReference[oaicite:1]{index=1}
from src.models.kaggle_metric import score as kaggle_score          # :contentReference[oaicite:2]{index=2}
from src.inference.postprocess import rle_encode                    # :contentReference[oaicite:3]{index=3}
from src.training.train_cv import build_solution_df                 # :contentReference[oaicite:4]{index=4}
from src.utils.seed_logging_utils import setup_seed, log_seed_info  # :contentReference[oaicite:5]{index=5}


def build_full_dataset(paths: Dict[str, str], img_size: int = 256) -> ForgeryDataset:
    """
    Full dataset with evaluation transform (no heavy aug).
    """
    transform = get_val_transform(img_size=img_size)
    ds = ForgeryDataset(
        transform=transform,
    )
    return ds


def load_model(weights_path: str, device: torch.device) -> Mask2FormerForgeryModel:
    """
    Create model with same hyperparams as train_full.py and load weights.
    """
    model = Mask2FormerForgeryModel(
        num_queries=15,
        d_model=256,
        authenticity_penalty_weight=5.0,
        auth_gate_forged_threshold=0.5,  # will be overridden per-sweep
        default_mask_threshold=0.5,
        default_cls_threshold=0.5,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference_for_gate(
    model: Mask2FormerForgeryModel,
    dataset: ForgeryDataset,
    solution_df: pd.DataFrame,
    gate: float,
    device: torch.device,
    batch_size: int = 4,
) -> float:
    """
    Run inference over the *whole training set* for a specific gate value,
    build a submission dataframe, and compute Kaggle metric.
    """
    # Update model's internal gate
    model.auth_gate_forged_threshold = float(gate)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    n_samples = len(dataset)
    oof_pred: List[str] = [None] * n_samples  # type: ignore

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)  # uses current auth_gate_forged_threshold

            for out, t in zip(outputs, targets):
                global_idx = int(t["image_id"].item())
                sample = dataset.samples[global_idx]
                img_path = sample["image_path"]
                mask_path = sample["mask_path"]

                # Original H, W as in train_cv.py
                if os.path.exists(mask_path):
                    m = np.load(mask_path)
                    if m.ndim == 3:
                        h, w = m.shape[1], m.shape[2]
                    else:
                        h, w = m.shape[0], m.shape[1]
                else:
                    with Image.open(img_path) as pil_img:
                        w, h = pil_img.size

                if out["masks"].numel() == 0:
                    pred_ann = "authentic"
                else:
                    # union all predicted masks at model resolution, upsample to original
                    union_small = (out["masks"] > 0).any(dim=0).float()  # [H', W']
                    union_small = union_small.unsqueeze(0).unsqueeze(0)  # [1,1,H',W']
                    union_up = (
                        F.interpolate(
                            union_small,
                            size=(h, w),
                            mode="nearest",
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    pred_ann = rle_encode(union_up)

                oof_pred[global_idx] = pred_ann

    # Build submission df aligned with solution_df
    submission = pd.DataFrame(
        {
            "row_id": solution_df["row_id"],
            "annotation": oof_pred,
        }
    )

    score = kaggle_score(
        solution_df.copy(),
        submission.copy(),
        row_id_column_name="row_id",
    )
    return float(score), submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep authenticity gate thresholds for a fixed full-train model "
            "and compute Kaggle metric on the training set."
        )
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

    # Model weights
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to full-train model weights (.pth)",
    )

    # Sweep config
    parser.add_argument(
        "--gates",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="List of authenticity gate thresholds to evaluate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Resize size used in validation transform (default: 256)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="experiments/auth_gate_sweep",
        help="Directory to store per-gate predictions and summary.",
    )

    # Seed / determinism (not critical for inference, but included for consistency)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no_deterministic",
        action="store_false",
        dest="deterministic",
        help="Disable deterministic mode.",
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    seed_info = setup_seed(args.seed, deterministic=args.deterministic)
    log_seed_info(seed_info)

    # Dataset + solution_df (ground truth)
    full_dataset = build_full_dataset(paths, img_size=args.img_size)
    print(f"Full dataset size: {len(full_dataset)}")

    solution_df, _ = build_solution_df(full_dataset)

    # Model
    model = load_model(args.weights, device=device)

    results = []

    for gate in args.gates:
        print(f"\n=== Evaluating auth_gate_forged_threshold = {gate:.3f} ===")
        score, submission = run_inference_for_gate(
            model=model,
            dataset=full_dataset,
            solution_df=solution_df,
            gate=gate,
            device=device,
            batch_size=args.batch_size,
        )
        print(f"Gate {gate:.3f} -> Kaggle metric (train/OOF-style): {score:.6f}")

        # Save per-gate predictions
        csv_path = out_dir / f"oof_predictions_gate_{gate:.3f}.csv"
        submission.to_csv(csv_path, index=False)
        print(f"Saved predictions to: {csv_path}")

        results.append(
            {
                "gate": float(gate),
                "score": float(score),
                "csv_path": str(csv_path),
            }
        )

    # Save summary
    summary_path = out_dir / "auth_gate_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep summary saved to: {summary_path}")
    print("Results (sorted by score):")
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        print(f"  gate={r['gate']:.3f}  score={r['score']:.6f}")


if __name__ == "__main__":
    main()
