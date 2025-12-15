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

from src.data.dataloader import ForgeryDataset, get_train_transform
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

def debug_optimizer_params(model, optimizer, keywords=("img_head", "class_head", "gate")):
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    named = list(model.named_parameters())

    total_opt = sum(p.numel() for p in model.parameters() if id(p) in opt_param_ids)
    print(f"[OPT DEBUG] total params in optimizer: {total_opt:,}")

    for kw in keywords:
        matched = [(n, p.numel()) for n, p in named if kw in n and id(p) in opt_param_ids]
        count = sum(n for _, n in matched)
        print(f"[OPT DEBUG] '{kw}' in optimizer: {bool(matched)} ({count:,} params)")
        if matched:
            print("   examples:", [n for n, _ in matched[:3]])


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
    num_epochs=30,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    debug_optimizer_params(model, optimizer)

    # Training loop
    print("Training on FULL DATASET:", len(train_dataset), "samples")

    printed_mask_stats = False
    printed_logit_stats = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # ---- debug: pick 1 fixed batch per epoch (train) ----
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
                t = targets[0]
                m = t["masks"]
                print("img_label:", int(t["image_label"].item()) if "image_label" in t else None,
                    "masks_shape:", tuple(m.shape),
                    "masks_sum:", float(m.sum().item()))
            # sanity block
            if not printed_mask_stats:
                with torch.no_grad():
                    for i, t in enumerate(targets):
                        m = t["masks"]
                        if m.numel() == 0:
                            print(f"[img {i}] mask: EMPTY")
                            continue

                        m = m.float()
                        print(
                            f"[img {i}] "
                            f"mean={m.mean(dim=(1,2)).tolist()} "
                            f"max={m.amax(dim=(1,2)).tolist()} "
                            f"frac>0.5={(m > 0.5).float().mean(dim=(1,2)).tolist()}"
                        )
                printed_mask_stats = True
            # ---- debug: model output stats for ONE batch (pre-loss) ----
            if not printed_logit_stats:
                with torch.no_grad():
                    mask_logits, class_logits, img_logits = model.forward_logits(images)

                    # logits stats (catch blow-ups)
                    ml = mask_logits
                    cl = class_logits
                    il = img_logits

                    print(
                        "[debug logits] "
                        f"mask_logits: min={ml.min().item():.4f} max={ml.max().item():.4f} "
                        f"mean={ml.mean().item():.4f} std={ml.std().item():.4f} | "
                        f"class_logits: min={cl.min().item():.4f} max={cl.max().item():.4f} "
                        f"mean={cl.mean().item():.4f} std={cl.std().item():.4f} | "
                        f"img_logits: min={il.min().item():.4f} max={il.max().item():.4f} "
                        f"mean={il.mean().item():.4f} std={il.std().item():.4f}"
                    )

                    # prob stats (catch double-sigmoid / saturation)
                    mask_probs = ml.sigmoid()
                    class_probs = cl.sigmoid()
                    img_probs = il.sigmoid()

                    # mask saturation diagnostics
                    print(
                        "[debug probs] "
                        f"mask_probs: mean={mask_probs.mean().item():.6f} "
                        f"p95={mask_probs.flatten().quantile(0.95).item():.6f} "
                        f"max={mask_probs.max().item():.6f} "
                        f"frac>0.5={(mask_probs > 0.5).float().mean().item():.6f} | "
                        f"class_probs: mean={class_probs.mean().item():.6f} "
                        f"max={class_probs.max().item():.6f} "
                        f"frac>0.1={(class_probs > 0.1).float().mean().item():.6f} | "
                        f"img_probs: mean={img_probs.mean().item():.6f} "
                        f"max={img_probs.max().item():.6f}"
                    )

                printed_logit_stats = True    

            loss_dict = model(images, targets)
            loss = loss_dict["loss_total"]

            w = dict(
                mask_bce=getattr(model, "loss_weight_mask_bce", None),
                mask_dice=getattr(model, "loss_weight_mask_dice", None),
                mask_cls=getattr(model, "loss_weight_mask_cls", None),
                img_auth=getattr(model, "loss_weight_img_auth", None),
                auth_pen=getattr(model, "loss_weight_auth_penalty", None),
                )

            print({
                "loss_mask_bce": float(loss_dict["loss_mask_bce"].detach().cpu()),
                "loss_mask_dice": float(loss_dict["loss_mask_dice"].detach().cpu()),
                "loss_mask_cls": float(loss_dict["loss_mask_cls"].detach().cpu()),
                "loss_img_auth": float(loss_dict["loss_img_auth"].detach().cpu()),
                "loss_auth_penalty": float(loss_dict["loss_auth_penalty"].detach().cpu()),
                "loss_total": float(loss_dict["loss_total"].detach().cpu()),
                })

            optimizer.zero_grad()
            loss.backward()

            def any_param(m):
                return next((p for p in m.parameters() if p.requires_grad), None)

            def grad_norm(p):
                return None if (p is None or p.grad is None) else p.grad.norm().item()

            # ---Debug---
            # with torch.no_grad():
            #     print(
            #         f"grad | img_head={grad_norm(any_param(model.img_head))} "
            #         f"cls_head={grad_norm(any_param(model.class_head))} "
            #         f"gate={grad_norm(any_param(model.auth_gate)) if hasattr(model, 'auth_gate') else None}"
            #     )
            # def has_grad(mod):
            #     ps = [p for p in mod.parameters() if p.requires_grad]
            #     return any(p.grad is not None for p in ps)

            # print("grads:",
            #     "class_head", has_grad(model.class_head),
            #     "mask_embed_head", has_grad(model.mask_embed_head),
            #     "img_head", has_grad(model.img_head))    
            optimizer.step()

            running_loss += loss.item() * len(images)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

        # --- wandb: per-epoch full-train loss ---
        log_epoch_metrics(
            stage="train",
            metrics={"loss": epoch_loss},
            epoch=epoch + 1,
            global_step=epoch + 1,
        )

        # ---- debug: logit/prob sanity (collapse detectors) ----
        model.eval()
        with torch.no_grad():
            mask_logits, class_logits, img_logits = model.forward_logits(debug_images)

            cls_probs = class_logits.sigmoid()                 # [B,Q]
            img_probs = img_logits.sigmoid()                   # [B]
            mask_probs = mask_logits.sigmoid().flatten(2)      # [B,Q,HW]

            cls_max = cls_probs.max(dim=1).values              # [B]
            mask_max = mask_probs.max(dim=2).values.max(dim=1).values  # [B] (max over Q and HW)

            dbg = {
                "cls_max_mean": cls_max.mean().item(),
                "cls_max_p95": cls_max.quantile(0.95).item(),
                "keep_rate@0.1": (cls_probs > 0.1).float().mean().item(),
                "keep_rate@0.2": (cls_probs > 0.2).float().mean().item(),
                "keep_rate@0.3": (cls_probs > 0.3).float().mean().item(),
                "img_forged_mean": img_probs.mean().item(),
                "mask_max_mean": mask_max.mean().item(),
            }

        log_epoch_metrics(
            stage="debug",
            metrics=dbg,
            epoch=epoch + 1,
            global_step=epoch + 1,
        )
        model.train()

    # Save weights
    torch.save(model.state_dict(), str(save_path))
    print(f"Model weights saved to: {save_path}")

    # --- wandb: final loss + model artifact ---
    set_summary("final_train_loss", float(epoch_loss))
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
