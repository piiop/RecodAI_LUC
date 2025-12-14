# src/training/train_cv.py

"""
Run K-fold cross-validation with Mask2FormerForgeryModel, producing per-fold and
overall OOF predictions scored with the official Kaggle metric. Builds the
solution dataframe, trains each fold, generates OOF masks/labels, computes
metrics, and saves CSV/JSON outputs under experiments/oof_results.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

from src.data.dataloader import (
    ForgeryDataset,
    get_train_transform,
    get_val_transform,
)
from src.inference.postprocess import rle_encode
from src.models.mask2former_v1 import Mask2FormerForgeryModel
from src.models.kaggle_metric import score as kaggle_score
from src.utils.seed_logging_utils import setup_seed, log_seed_info
from src.utils.wandb_utils import (
    init_wandb_run,
    log_config,
    log_epoch_metrics,
    set_summary,
    log_best_metric,
    finish_run,
)


def build_solution_df(full_dataset):
    """
    Build the Kaggle-style 'solution_df' from the full training dataset.

    Expects full_dataset.samples[i] dicts with keys:
        - 'image_id'
        - 'is_forged'
        - 'image_path'
        - 'mask_path'
    """
    n_samples = len(full_dataset)

    image_ids = [full_dataset.samples[i]["image_id"] for i in range(n_samples)]
    is_forged = np.array(
        [1 if full_dataset.samples[i]["is_forged"] else 0 for i in range(n_samples)]
    )

    gt_annotations = []
    gt_shapes = []

    for i in range(n_samples):
        sample = full_dataset.samples[i]
        img_path = sample["image_path"]
        mask_path = sample["mask_path"]

        # Default shape from image
        with Image.open(img_path) as pil_img:
            w, h = pil_img.size

        if sample["is_forged"] and os.path.exists(mask_path):
            m = np.load(mask_path)

            # Same handling as in the notebook/dataset
            if m.ndim == 3 and m.shape[0] <= 10 and m.shape[1:] == (h, w):
                m = np.any(m, axis=0)
            m = (m > 0).astype(np.uint8)

            gt_annotations.append(rle_encode(m))
        else:
            gt_annotations.append("authentic")

        gt_shapes.append(json.dumps([h, w]))  # e.g. "[720, 960]"

    solution_df = pd.DataFrame(
        {
            "row_id": image_ids,
            "annotation": gt_annotations,
            "shape": gt_shapes,
        }
    )

    return solution_df, is_forged


def make_datasets(train_transform=None, val_transform=None):
    """
    Helper to build the full_dataset (no transforms) + template train/val datasets.
    """
    full_dataset = ForgeryDataset(
        transform=None,
    )

    ds_train = ForgeryDataset(
        transform=train_transform,
    )

    ds_val = ForgeryDataset(
        transform=val_transform,
    )

    return full_dataset, ds_train, ds_val


def run_cv(
    num_folds=5,
    num_epochs=3,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    device=None,
    train_transform=None,
    val_transform=None,
    out_dir="experiments/oof_results",
    model_cfg=None,
    model_kwargs=None,     
):
    """
    Main K-fold CV + OOF routine.

    Args:
        num_folds, num_epochs, batch_size, lr, weight_decay: training hyperparams
        device: torch.device or None (auto)
        train_transform, val_transform: optional transforms to pass to ForgeryDataset
        out_dir: where to save OOF CSVs / metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(out_dir, exist_ok=True)

    # --- plug in default Albumentations transforms if not provided ---
    if train_transform is None:
        train_transform = get_train_transform()
    if val_transform is None:
        val_transform = get_val_transform()

    # Build datasets
    full_dataset, ds_train_template, ds_val_template = make_datasets(
        train_transform=train_transform, val_transform=val_transform
    )
    solution_df, is_forged = build_solution_df(full_dataset)

    n_samples = len(full_dataset)
    print(f"Built solution_df with {len(solution_df)} rows")

    # CV splitter
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    oof_pred = [None] * n_samples
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(n_samples), is_forged)
    ):
        print(f"\n===== Fold {fold + 1}/{num_folds} =====")
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()

        # Fresh datasets per fold so transforms are independent if they are stateful
        ds_train = ds_train_template  # same underlying data; subset controls indices
        ds_val = ds_val_template

        train_loader = DataLoader(
            torch.utils.data.Subset(ds_train, train_idx),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=4,          # try 4 first; can test 8 later
            pin_memory=True,        
            persistent_workers=True 
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(ds_val, val_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        # Model + optimizer per fold
        mk = {} if model_kwargs is None else dict(model_kwargs)
        if "d_model" not in mk:
            mk["d_model"] = 256  

        model = Mask2FormerForgeryModel(**mk).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # ---- Train ----
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, targets in train_loader:
                images = [img.to(device) for img in images]

                for t in targets:
                    t["masks"] = t["masks"].to(device)
                    t["image_label"] = t["image_label"].to(device)

                loss_dict = model(images, targets)
                loss = loss_dict["loss_total"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(images)

            running_loss /= len(train_idx)
            print(
                f"Fold {fold + 1} Epoch {epoch + 1}/{num_epochs} - "
                f"train loss: {running_loss:.4f}"
            )

            # --- wandb: per-epoch train loss for this fold ---
            log_epoch_metrics(
                stage=f"fold{fold + 1}/train",
                metrics={"loss": running_loss},
                epoch=epoch + 1,
                global_step=fold * num_epochs + epoch + 1,
            )

            # ===================== DEBUG LOGITS (ADD THIS) =====================
            model.eval()
            with torch.no_grad():
                dbg_images, _ = next(iter(val_loader))
                dbg_images = [img.to(device) for img in dbg_images]

                images_t = torch.stack(dbg_images)

                fpn_feats = model.backbone(images_t)
                mask_feats = model.mask_feature_proj(fpn_feats[0])
                pos_list = [model.position_encoding(x) for x in fpn_feats]
                hs = model.transformer_decoder(fpn_feats, pos_list)[-1]

                class_logits = model.class_head(hs).squeeze(-1)   # [B, Q]
                mask_logits = torch.einsum(
                    "bqc,bchw->bqhw",
                    model.mask_embed_head(hs),
                    mask_feats,
                )
                img_logits = model.img_head(fpn_feats[-1]).squeeze(-1)

                cls_probs = class_logits.sigmoid()
                mask_probs = mask_logits.sigmoid()
                img_probs = img_logits.sigmoid()

                dbg_metrics = {
                    "cls/max_mean": cls_probs.max(dim=1).values.mean().item(),
                    "cls/keep@0.2": (cls_probs > 0.2).float().mean().item(),
                    "cls/keep@0.3": (cls_probs > 0.3).float().mean().item(),
                    "mask/max_mean": mask_probs.flatten(2).max(dim=2).values.mean().item(),
                    "img/forged_mean": img_probs.mean().item(),
                }

            log_epoch_metrics(
                stage=f"fold{fold + 1}/debug",
                metrics=dbg_metrics,
                epoch=epoch + 1,
                global_step=fold * num_epochs + epoch + 1,
            )

            model.train()

        # ---- Inference on validation fold (OOF) ----
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)  # calls model.inference(...)

                for out, t in zip(outputs, targets):
                    # Global dataset index (same as in full_dataset.samples)
                    global_idx = int(t["image_id"].item())
                    sample = full_dataset.samples[global_idx]
                    img_path = sample["image_path"]
                    mask_path = sample["mask_path"]

                    # Original H, W
                    if os.path.exists(mask_path):
                        m = np.load(mask_path)
                        if m.ndim == 3:
                            h, w = m.shape[1], m.shape[2]
                        else:
                            h, w = m.shape[0], m.shape[1]
                    else:
                        with Image.open(img_path) as pil_img:
                            w, h = pil_img.size

                    # Convert model outputs to competition-style annotation
                    if out["masks"].numel() == 0:
                        # authenticity gate said authentic
                        pred_ann = "authentic"
                    else:
                        # union all predicted masks then upsample to original size
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

        # ---- Fold metric using official score ----
        fold_solution = solution_df.iloc[val_idx].reset_index(drop=True)
        fold_submission = pd.DataFrame(
            {
                "row_id": fold_solution["row_id"],
                "annotation": [oof_pred[i] for i in val_idx],
            }
        )

        fold_score = kaggle_score(
            fold_solution.copy(),
            fold_submission.copy(),
            row_id_column_name="row_id",
        )
        fold_scores.append(fold_score)
        print(f"Fold {fold + 1} metric: {fold_score:.6f}")

        # --- wandb: per-fold validation metric ---
        log_epoch_metrics(
            stage=f"fold{fold + 1}/val",
            metrics={"kaggle_metric": fold_score},
            epoch=num_epochs,
            global_step=(fold + 1) * num_epochs,
        )

        # Optionally save per-fold predictions
        fold_submission.to_csv(
            os.path.join(out_dir, f"fold_{fold + 1}_oof.csv"), index=False
        )

    # --- Overall OOF metric ---
    oof_solution = solution_df.copy()
    oof_submission = pd.DataFrame(
        {
            "row_id": oof_solution["row_id"],
            "annotation": oof_pred,
        }
    )

    oof_score = kaggle_score(
        oof_solution.copy(),
        oof_submission.copy(),
        row_id_column_name="row_id",
    )

    mean_cv = float(np.mean(fold_scores))
    print("\nPer-fold scores:", fold_scores)
    print("Mean CV:", mean_cv)
    print("OOF score:", float(oof_score))

    # Save overall OOF predictions + metrics
    oof_submission.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    metrics = {
        "fold_scores": [float(s) for s in fold_scores],
        "mean_cv": mean_cv,
        "oof_score": float(oof_score),
    }
    with open(os.path.join(out_dir, "oof_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- wandb: summary fields for this CV run ---
    set_summary("fold_scores", metrics["fold_scores"])
    set_summary("mean_cv", metrics["mean_cv"])
    set_summary("oof_score", metrics["oof_score"])
    log_best_metric("mean_cv", metrics["mean_cv"], higher_is_better=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="K-fold CV + OOF training for Mask2FormerForgeryModel"
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs per fold (default: 3)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
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

    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    seed_info = setup_seed(args.seed, deterministic=args.deterministic)
    log_seed_info(seed_info)

    # --- wandb: init run + log config ---
    run = init_wandb_run(
        config=vars(args),
        project="mask2former-forgery",
        job_type="cv",
        group="cv",
        name=f"cv_{Path(args.out_dir).name}",
    )

    try:
        run_cv(
            num_folds=args.n_folds,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            train_transform=None,  # plug in your train transform here if desired
            val_transform=None,  # plug in your val transform here if desired
            out_dir=args.out_dir,
        )
    finally:
        finish_run()

if __name__ == "__main__":
    main()
