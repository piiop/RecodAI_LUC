# src/training/train_cv.py

"""
Run K-fold cross-validation with Mask2FormerForgeryModel, producing per-fold and
overall OOF predictions scored with the official Kaggle metric.

Aligned with train_full:
- Uses ClsCollapseLogger (JSON/CSV/JSONL debug logs)
- Uses model(..., inference_overrides={"logger":..., "debug_ctx":...}) during training
- Uses model.forward_logits(...) for debug stats (no stale internal calls)
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

from src.data.dataloader import (
    ForgeryDataset,
    get_train_transform,
    get_val_transform,
    make_groupkfold_splits
)
from src.inference.postprocess import rle_encode
from src.models.mask2former_v1 import Mask2FormerForgeryModel
from src.models.kaggle_metric import score as kaggle_score
from src.utils.seed_logging_utils import setup_seed, log_seed_info
from src.utils.wandb_utils import (
    init_wandb_run,
    log_epoch_metrics,
    set_summary,
    log_best_metric,
    finish_run,
)
from src.utils.config_utils import sanitize_model_kwargs
from src.utils.cls_collapse_logger import ClsCollapseLogger


def collect_optimizer_debug(model, optimizer, keywords=("img_head", "class_head", "gate")):
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    named = list(model.named_parameters())

    total_opt = sum(p.numel() for p in model.parameters() if id(p) in opt_param_ids)

    out = {"total_params_in_optimizer": int(total_opt), "keywords": {}}
    for kw in keywords:
        matched = [(n, p.numel()) for n, p in named if kw in n and id(p) in opt_param_ids]
        out["keywords"][kw] = {
            "present": bool(matched),
            "params": int(sum(num for _, num in matched)),
            "examples": [n for n, _ in matched[:3]],
        }
    return out


def build_solution_df(dataset):
    """
    Build Kaggle-style solution_df for the FULL training dataset.

    - row_id: image basename stem (matches mask naming convention)
    - annotation: RLE of union mask, or "authentic"
    - shape: JSON "[H, W]" from the original image size

    Also returns `y_strat`: 0/1 labels aligned with ForgeryDataset.__getitem__:
      forged only if (is_forged == True) AND (mask has any positive pixel).
    """
    gt_rows = []
    y_strat = np.zeros(len(dataset), dtype=np.int64)

    for idx, sample in enumerate(dataset.samples):
        img_path = sample["image_path"]
        mask_path = sample.get("mask_path", None)
        row_id = Path(img_path).stem

        # Original image shape (Kaggle expects original H,W)
        with Image.open(img_path) as im:
            w, h = im.size

        ann = "authentic"
        forged_effective = False

        if sample.get("is_forged", False) and mask_path and Path(mask_path).exists():
            m = np.load(mask_path)

            # allow either (H,W) or (K,H,W); union instances if needed
            if m.ndim == 3:
                m = np.any(m > 0, axis=0).astype(np.uint8)
            else:
                m = (m > 0).astype(np.uint8)

            if m.sum() > 0:
                ann = rle_encode(m)
                forged_effective = True

        y_strat[idx] = 1 if forged_effective else 0
        gt_rows.append(
            {"row_id": row_id, "annotation": ann, "shape": json.dumps([h, w])}
        )

    solution_df = pd.DataFrame(gt_rows)
    return solution_df, y_strat


def make_datasets(train_transform=None, val_transform=None):
    """
    Build:
      - full_dataset (transform=None) used for solution_df + sample paths
      - ds_train (with train_transform)
      - ds_val (with val_transform)
    """
    full_dataset = ForgeryDataset(transform=None)
    ds_train = ForgeryDataset(transform=train_transform)
    ds_val = ForgeryDataset(transform=val_transform)
    return full_dataset, ds_train, ds_val


def collate_batch(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _move_targets_to_device(targets, device):
    for t in targets:
        if "masks" in t:
            t["masks"] = t["masks"].to(device)
        if "image_label" in t:
            t["image_label"] = t["image_label"].to(device)


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
    model_kwargs=None,
    debug_out_dir="experiments/cls_collapse",
    enable_debug_logs=True,
):
    """
    Main K-fold CV + OOF routine.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(out_dir, exist_ok=True)

    if train_transform is None:
        train_transform = get_train_transform()
    if val_transform is None:
        val_transform = get_val_transform()

    full_dataset, ds_train, ds_val = make_datasets(
        train_transform=train_transform, val_transform=val_transform
    )
    solution_df, is_forged = build_solution_df(full_dataset)

    n_samples = len(full_dataset)
    print(f"Built solution_df with {len(solution_df)} rows")

    # GroupKFold splits to prevent leakage across authentic/forged variants
    splits = make_groupkfold_splits(full_dataset, n_splits=num_folds)

    oof_pred = [None] * n_samples
    fold_scores = []

    mk = sanitize_model_kwargs(model_kwargs or {})

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n===== Fold {fold + 1}/{num_folds} =====")
        if hasattr(train_idx, "tolist"):
            train_idx = train_idx.tolist()
        if hasattr(val_idx, "tolist"):
            val_idx = val_idx.tolist()

        train_loader = DataLoader(
            torch.utils.data.Subset(ds_train, train_idx),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(ds_val, val_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        model = Mask2FormerForgeryModel(**mk).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # fold logger (mirrors train_full)
        run_name = f"cv_fold{fold + 1}"
        collapse_logger = ClsCollapseLogger(
            out_dir=debug_out_dir,
            run_name=run_name,
            enable_debug=bool(enable_debug_logs),
        )
        collapse_logger.write_meta(
            {
                "fold": fold + 1,
                "num_folds": num_folds,
                "device": str(device),
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "out_dir": str(out_dir),
                "model_kwargs": mk,
                "train_len": len(train_idx),
                "val_len": len(val_idx),
            }
        )
        collapse_logger.write_optimizer_debug(collect_optimizer_debug(model, optimizer))

        # fixed debug batch each epoch (from val)
        dbg_images, dbg_targets = next(iter(val_loader))
        dbg_images = [img.to(device) for img in dbg_images]
        _move_targets_to_device(dbg_targets, device)

        global_step = 0
        printed_mask_stats = False
        printed_logit_stats = False

        # ---- Train ----
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                _move_targets_to_device(targets, device)

                # lightweight batch target sanity
                t0 = targets[0]
                m0 = t0.get("masks", None)
                collapse_logger.debug_event(
                    "batch_target0",
                    {
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "img_label": int(t0["image_label"].item()) if "image_label" in t0 else None,
                        "masks_shape": list(m0.shape) if m0 is not None else None,
                        "masks_sum": float(m0.sum().item()) if (m0 is not None and m0.numel()) else 0.0,
                    },
                )

                # one-time mask stats
                if not printed_mask_stats:
                    payload = {"fold": fold + 1, "epoch": epoch + 1, "global_step": global_step, "per_image": []}
                    with torch.no_grad():
                        for i, t in enumerate(targets):
                            m = t.get("masks", None)
                            if m is None or m.numel() == 0:
                                payload["per_image"].append({"i": i, "empty": True})
                                continue
                            mf = m.float()
                            payload["per_image"].append(
                                {
                                    "i": i,
                                    "empty": False,
                                    "mean_per_inst": mf.mean(dim=(1, 2)).tolist(),
                                    "max_per_inst": mf.amax(dim=(1, 2)).tolist(),
                                    "frac_gt_0p5_per_inst": (mf > 0.5).float().mean(dim=(1, 2)).tolist(),
                                }
                            )
                    collapse_logger.debug_event("mask_target_sanity", payload)
                    printed_mask_stats = True

                # one-time logits/probs stats
                if not printed_logit_stats:
                    with torch.no_grad():
                        mask_logits, class_logits, img_logits = model.forward_logits(images)
                        mask_probs = mask_logits.sigmoid()
                        class_probs = class_logits.sigmoid()
                        img_probs = img_logits.sigmoid()

                        collapse_logger.debug_event(
                            "debug_probs",
                            {
                                "fold": fold + 1,
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "mask_probs": {
                                    "mean": mask_probs.mean().item(),
                                    "p95": mask_probs.flatten().quantile(0.95).item(),
                                    "max": mask_probs.max().item(),
                                    "frac_gt_0p5": (mask_probs > 0.5).float().mean().item(),
                                },
                                "class_probs": {
                                    "mean": class_probs.mean().item(),
                                    "max": class_probs.max().item(),
                                    "frac_gt_0p1": (class_probs > 0.1).float().mean().item(),
                                },
                                "img_probs": {"mean": img_probs.mean().item(), "max": img_probs.max().item()},
                            },
                        )
                    printed_logit_stats = True

                loss_dict = model(
                    images,
                    targets,
                    inference_overrides={
                        "logger": collapse_logger,
                        "debug_ctx": {"fold": fold + 1, "epoch": epoch + 1, "global_step": global_step},
                    },
                )
                loss = loss_dict["loss_total"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                collapse_logger.log_step_losses(
                    {
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss_mask_bce": float(loss_dict["loss_mask_bce"].detach().cpu()),
                        "loss_mask_dice": float(loss_dict["loss_mask_dice"].detach().cpu()),
                        "loss_mask_cls": float(loss_dict["loss_mask_cls"].detach().cpu()),
                        "loss_img_auth": float(loss_dict["loss_img_auth"].detach().cpu()),
                        "loss_auth_penalty": float(loss_dict["loss_auth_penalty"].detach().cpu()),
                        "loss_total": float(loss_dict["loss_total"].detach().cpu()),
                        "w_mask_cls": float(getattr(model, "loss_weight_mask_cls", 0.0)),
                        "w_auth_penalty": float(getattr(model, "loss_weight_auth_penalty", 0.0)),
                    }
                )

                running_loss += loss.item() * len(images)
                global_step += 1

            train_epoch_loss = running_loss / max(len(train_idx), 1)
            print(
                f"Fold {fold + 1} Epoch {epoch + 1}/{num_epochs} - "
                f"train loss: {train_epoch_loss:.4f}"
            )
            log_epoch_metrics(
                stage=f"fold{fold + 1}/train",
                metrics={"loss": train_epoch_loss},
                epoch=epoch + 1,
                global_step=fold * num_epochs + epoch + 1,
            )

            # epoch-end collapse stats (on fixed debug batch)
            model.eval()
            with torch.no_grad():
                mask_logits, class_logits, img_logits = model.forward_logits(dbg_images)
                cls_probs = class_logits.sigmoid()
                img_probs = img_logits.sigmoid()
                mask_probs = mask_logits.sigmoid().flatten(2)

                cls_max = cls_probs.max(dim=1).values
                mask_max = mask_probs.max(dim=2).values.max(dim=1).values

                collapse_logger.log_epoch_summary(
                    {
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "epoch_loss": float(train_epoch_loss),
                        "cls_max_mean": cls_max.mean().item(),
                        "cls_max_p95": cls_max.quantile(0.95).item(),
                        "keep_rate@0.1": (cls_probs > 0.1).float().mean().item(),
                        "keep_rate@0.2": (cls_probs > 0.2).float().mean().item(),
                        "keep_rate@0.3": (cls_probs > 0.3).float().mean().item(),
                        "img_forged_mean": img_probs.mean().item(),
                        "mask_max_mean": mask_max.mean().item(),
                        "w_mask_cls": float(getattr(model, "loss_weight_mask_cls", 0.0)),
                    }
                )

                log_epoch_metrics(
                    stage=f"fold{fold + 1}/debug",
                    metrics={
                        "cls/max_mean": cls_max.mean().item(),
                        "cls/keep@0.2": (cls_probs > 0.2).float().mean().item(),
                        "cls/keep@0.3": (cls_probs > 0.3).float().mean().item(),
                        "mask/max_mean": mask_max.mean().item(),
                        "img/forged_mean": img_probs.mean().item(),
                    },
                    epoch=epoch + 1,
                    global_step=fold * num_epochs + epoch + 1,
                )

            model.train()

        # ---- Inference on validation fold (OOF) ----
        model.eval()
        with torch.no_grad():

            # unified inference-debug stats (aggregated over the whole val fold)
            inf_dbg = {
                "n": 0,
                "masks_empty": 0,
                "gate_fail": 0,
                "num_keep0": 0,
                "cls_filtered_all_fg": 0,  # any_fg_pre_keep=True but any_fg_post_keep=False
                "max_cls_prob": [],
                "max_mask_prob": [],
            }

            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)  # inference path

                for out, t in zip(outputs, targets):
                    inf_dbg["n"] += 1

                    # counts
                    if out["masks"].numel() == 0:
                        inf_dbg["masks_empty"] += 1
                    if out.get("gate_pass") is False:
                        inf_dbg["gate_fail"] += 1
                    if int(out.get("num_keep", 0)) == 0:
                        inf_dbg["num_keep0"] += 1
                    if bool(out.get("any_fg_pre_keep", False)) and (not bool(out.get("any_fg_post_keep", False))):
                        inf_dbg["cls_filtered_all_fg"] += 1

                    # collect scalars (already returned by model.inference)
                    inf_dbg["max_cls_prob"].append(float(out.get("max_cls_prob", 0.0)))
                    inf_dbg["max_mask_prob"].append(float(out.get("max_mask_prob", 0.0)))

                    global_idx = int(t["image_id"].item())
                    sample = full_dataset.samples[global_idx]
                    img_path = sample["image_path"]
                    mask_path = sample["mask_path"]

                    # original H,W
                    if mask_path and os.path.exists(mask_path):
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
                        union_small = (out["masks"] > 0).any(dim=0).float()[None, None]
                        union_up = (
                            F.interpolate(union_small, size=(h, w), mode="nearest")
                            .squeeze()
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                        pred_ann = rle_encode(union_up)

                    oof_pred[global_idx] = pred_ann

        # ---- Fold inference debug summary (single structured log) ----
        n = max(int(inf_dbg["n"]), 1)

        max_cls = torch.tensor(inf_dbg["max_cls_prob"], dtype=torch.float32) if inf_dbg["max_cls_prob"] else torch.tensor([0.0])
        max_msk = torch.tensor(inf_dbg["max_mask_prob"], dtype=torch.float32) if inf_dbg["max_mask_prob"] else torch.tensor([0.0])

        collapse_logger.debug_event(
            "oof_inference_debug",
            {
                "fold": fold + 1,
                "val_samples": int(inf_dbg["n"]),
                "masks_empty": int(inf_dbg["masks_empty"]),
                "gate_fail": int(inf_dbg["gate_fail"]),
                "num_keep0": int(inf_dbg["num_keep0"]),
                "cls_filtered_all_fg": int(inf_dbg["cls_filtered_all_fg"]),
                "rates": {
                    "masks_empty": inf_dbg["masks_empty"] / n,
                    "gate_fail": inf_dbg["gate_fail"] / n,
                    "num_keep0": inf_dbg["num_keep0"] / n,
                    "cls_filtered_all_fg": inf_dbg["cls_filtered_all_fg"] / n,
                },
                "max_cls_prob": {
                    "mean": float(max_cls.mean().item()),
                    "p95": float(max_cls.quantile(0.95).item()),
                    "max": float(max_cls.max().item()),
                },
                "max_mask_prob": {
                    "mean": float(max_msk.mean().item()),
                    "p95": float(max_msk.quantile(0.95).item()),
                    "max": float(max_msk.max().item()),
                },
            },
        )

        # ---- Fold metric ----
        fold_solution = solution_df.iloc[val_idx].reset_index(drop=True)
        fold_submission = pd.DataFrame(
            {"row_id": fold_solution["row_id"], "annotation": [oof_pred[i] for i in val_idx]}
        )

        fold_score = kaggle_score(
            fold_solution.copy(),
            fold_submission.copy(),
            row_id_column_name="row_id",
        )
        fold_scores.append(float(fold_score))
        print(f"Fold {fold + 1} metric: {float(fold_score):.6f}")

        log_epoch_metrics(
            stage=f"fold{fold + 1}/val",
            metrics={"kaggle_metric": float(fold_score)},
            epoch=num_epochs,
            global_step=(fold + 1) * num_epochs,
        )

        fold_submission.to_csv(os.path.join(out_dir, f"fold_{fold + 1}_oof.csv"), index=False)

    # ---- Overall OOF ----
    oof_solution = solution_df.copy()
    oof_submission = pd.DataFrame({"row_id": oof_solution["row_id"], "annotation": oof_pred})

    oof_score = kaggle_score(
        oof_solution.copy(),
        oof_submission.copy(),
        row_id_column_name="row_id",
    )

    mean_cv = float(np.mean(fold_scores))
    print("\nPer-fold scores:", fold_scores)
    print("Mean CV:", mean_cv)
    print("OOF score:", float(oof_score))

    oof_submission.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    metrics = {"fold_scores": [float(s) for s in fold_scores], "mean_cv": mean_cv, "oof_score": float(oof_score)}
    with open(os.path.join(out_dir, "oof_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    set_summary("fold_scores", metrics["fold_scores"])
    set_summary("mean_cv", metrics["mean_cv"])
    set_summary("oof_score", metrics["oof_score"])
    log_best_metric("mean_cv", metrics["mean_cv"], higher_is_better=True)


def parse_args():
    parser = argparse.ArgumentParser(description="K-fold CV + OOF training for Mask2FormerForgeryModel")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="experiments/oof_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.set_defaults(deterministic=True)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    seed_info = setup_seed(args.seed, deterministic=args.deterministic)
    log_seed_info(seed_info)

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
            train_transform=None,
            val_transform=None,
            out_dir=args.out_dir,
            model_kwargs=None,
        )
    finally:
        finish_run()


if __name__ == "__main__":
    main()
