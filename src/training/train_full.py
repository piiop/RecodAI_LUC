# src/training/train_full.py

"""
Train a Mask2FormerForgeryModel on the **entire** training set using the
best hyperparameters discovered via CV.  
Outputs final weights for Kaggle LB evaluation (stored under weights/full_train/).
"""

import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataloader import ForgeryDataset, get_train_transform, make_groupkfold_splits

from src.models.mask2former_v1 import Mask2FormerForgeryModel
from src.utils.seed_logging_utils import setup_seed, log_seed_info
from src.utils.wandb_utils import (
    init_wandb_run,
    log_config,
    log_epoch_metrics,
    set_summary,
    log_artifact,
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

def build_train_dataset(train_transform=None):
    """
    Full training dataset (same data as CV, with chosen train transform).
    """
    dataset = ForgeryDataset(
        transform=train_transform,
    )
    return dataset

def collate_batch(batch):
    """
    Simple collate function for (image, target) pairs.
    """
    images, targets = zip(*batch)  # batch is a list of (img, target)
    return list(images), list(targets)

def run_full_train(
    num_epochs=25,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    device=None,
    train_transform=None,
    save_path="weights/full_train/model_full_data_baseline.pth",
    model_kwargs=None,
):
    """
    Train Mask2FormerForgeryModel on the FULL dataset.

    Args:
        paths: dict with paths to train/supp data (same as in CV script)
        num_epochs, batch_size, lr, weight_decay: training hyperparams
        device: torch.device or None (auto)
        train_transform: transform to apply (should match CV train transform)
        save_path: where to save final model weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure save directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Full dataset with same transforms as CV training
    train_dataset = build_train_dataset(train_transform=train_transform)

    if train_transform is None:
        train_transform = get_train_transform()
    train_dataset = build_train_dataset(train_transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=4,          # try 4 first; can test 8 later
        pin_memory=True,        
        persistent_workers=True 
    )

    mk = {} if model_kwargs is None else dict(model_kwargs)
    if "d_model" not in mk:
        mk["d_model"] = 256  

    mk = sanitize_model_kwargs(model_kwargs)
    model = Mask2FormerForgeryModel(
        **mk,
    ).to(device)

    run_name = f"full_{save_path.stem}"
    collapse_logger = ClsCollapseLogger(out_dir="experiments/cls_collapse", run_name=run_name)
    collapse_logger.write_meta({
        "device": str(device),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "save_path": str(save_path),
        "model_kwargs": mk,
        "dataset_len": len(train_dataset),
    })

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    collapse_logger.write_optimizer_debug(collect_optimizer_debug(model, optimizer))

    # Training loop
    print("Training on FULL DATASET:", len(train_dataset), "samples")

    printed_mask_stats = False
    printed_logit_stats = False
    global_step = 0
    epoch_loss = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # fixed debug batch per epoch
        debug_images, debug_targets = next(iter(train_loader))
        debug_images = [img.to(device) for img in debug_images]
        for t in debug_targets:
            t["masks"] = t["masks"].to(device)
            t["image_label"] = t["image_label"].to(device)

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            for t in targets:
                t["masks"] = t["masks"].to(device)
                t["image_label"] = t["image_label"].to(device)

            # ---- (was print of target[0] stats) ----
            t0 = targets[0]
            m0 = t0["masks"]
            collapse_logger.debug_event("batch_target0", {
                "epoch": epoch + 1,
                "global_step": global_step,
                "img_label": int(t0["image_label"].item()) if "image_label" in t0 else None,
                "masks_shape": list(m0.shape),
                "masks_sum": float(m0.sum().item()),
            })

            # ---- (was one-time sanity block) ----
            if not printed_mask_stats:
                payload = {"epoch": epoch + 1, "global_step": global_step, "per_image": []}
                with torch.no_grad():
                    for i, t in enumerate(targets):
                        m = t["masks"]
                        if m.numel() == 0:
                            payload["per_image"].append({"i": i, "empty": True})
                            continue
                        mf = m.float()
                        payload["per_image"].append({
                            "i": i,
                            "empty": False,
                            "mean_per_inst": mf.mean(dim=(1,2)).tolist(),
                            "max_per_inst": mf.amax(dim=(1,2)).tolist(),
                            "frac_gt_0p5_per_inst": (mf > 0.5).float().mean(dim=(1,2)).tolist(),
                        })
                collapse_logger.debug_event("mask_target_sanity", payload)
                printed_mask_stats = True

            # ---- (was one-time logits/probs stats pre-loss) ----
            if not printed_logit_stats:
                with torch.no_grad():
                    mask_logits, class_logits, img_logits = model.forward_logits(images)
                    ml, cl, il = mask_logits, class_logits, img_logits

                    mask_probs = ml.sigmoid()
                    class_probs = cl.sigmoid()
                    img_probs = il.sigmoid()

                    collapse_logger.debug_event("debug_logits", {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "mask_logits": {
                            "min": ml.min().item(), "max": ml.max().item(),
                            "mean": ml.mean().item(), "std": ml.std().item(),
                        },
                        "class_logits": {
                            "min": cl.min().item(), "max": cl.max().item(),
                            "mean": cl.mean().item(), "std": cl.std().item(),
                        },
                        "img_logits": {
                            "min": il.min().item(), "max": il.max().item(),
                            "mean": il.mean().item(), "std": il.std().item(),
                        },
                    })

                    collapse_logger.debug_event("debug_probs", {
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
                        "img_probs": {
                            "mean": img_probs.mean().item(),
                            "max": img_probs.max().item(),
                        },
                    })

                printed_logit_stats = True

            loss_dict = model(
                images,
                targets,
                inference_overrides={
                    "logger": collapse_logger,
                    "debug_ctx": {"epoch": epoch + 1, "global_step": global_step},
                },
            )
            loss = loss_dict["loss_total"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- (was print of loss dict) ----
            collapse_logger.log_step_losses({
                "epoch": epoch + 1,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "loss_mask_bce": float(loss_dict["loss_mask_bce"].detach().cpu()),
                "loss_mask_dice": float(loss_dict["loss_mask_dice"].detach().cpu()),
                "loss_mask_cls": float(loss_dict["loss_mask_cls"].detach().cpu()),
                "loss_img_auth": float(loss_dict["loss_img_auth"].detach().cpu()),
                "loss_auth_penalty": float(loss_dict["loss_auth_penalty"].detach().cpu()),
                "loss_total": float(loss_dict["loss_total"].detach().cpu()),
                # weights snapshot (helps sweep analysis)
                "w_mask_bce": float(getattr(model, "loss_weight_mask_bce", 0.0)),
                "w_mask_dice": float(getattr(model, "loss_weight_mask_dice", 0.0)),
                "w_mask_cls": float(getattr(model, "loss_weight_mask_cls", 0.0)),
                "w_img_auth": float(getattr(model, "loss_weight_img_auth", 0.0)),
                "w_auth_penalty": float(getattr(model, "loss_weight_auth_penalty", 0.0)),
            })

            running_loss += loss.item() * len(images)
            global_step += 1

        epoch_loss = running_loss / len(train_dataset)

        # ---- epoch-end collapse detectors (already existed) ----
        model.eval()
        with torch.no_grad():
            mask_logits, class_logits, img_logits = model.forward_logits(debug_images)

            cls_probs = class_logits.sigmoid()
            img_probs = img_logits.sigmoid()
            mask_probs = mask_logits.sigmoid().flatten(2)

            cls_max = cls_probs.max(dim=1).values
            mask_max = mask_probs.max(dim=2).values.max(dim=1).values

            collapse_logger.log_epoch_summary({
                "epoch": epoch + 1,
                "epoch_loss": float(epoch_loss),
                "cls_max_mean": cls_max.mean().item(),
                "cls_max_p95": cls_max.quantile(0.95).item(),
                "keep_rate@0.1": (cls_probs > 0.1).float().mean().item(),
                "keep_rate@0.2": (cls_probs > 0.2).float().mean().item(),
                "keep_rate@0.3": (cls_probs > 0.3).float().mean().item(),
                "img_forged_mean": img_probs.mean().item(),
                "mask_max_mean": mask_max.mean().item(),
                "w_mask_cls": float(getattr(model, "loss_weight_mask_cls", 0.0)),
            })

        model.train()

    torch.save(model.state_dict(), str(save_path))

    set_summary("final_train_loss", float(epoch_loss if epoch_loss is not None else 0.0))
    log_artifact(
        path=str(save_path),
        name=save_path.stem,
        type="model",
        aliases=["full_train", "latest"],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-data training for Mask2FormerForgeryModel "
        "(for Kaggle LB submission weights)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs (default: 30, use best from CV)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4, use best from CV)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4, use best from CV)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4, use best from CV)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="weights/full_train/model_full_data_baseline.pth",
        help="Path to save model weights (default: weights/full_train/model_full_data_baseline.pth)",
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

    seed_info = setup_seed(args.seed, deterministic=args.deterministic)
    log_seed_info(seed_info)

    # --- wandb: init run + log config ---
    run = init_wandb_run(
        config=vars(args),
        project="mask2former-forgery",
        job_type="full_train",
        group="full_train",
        name=f"full_{Path(args.save_path).stem}",
    )

    try:
        run_full_train(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            train_transform=get_train_transform(),
            save_path=args.save_path,
        )
    finally:
        finish_run()

if __name__ == "__main__":
    main()
