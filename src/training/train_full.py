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

from src.data.dataloader import ForgeryDataset
from src.models.mask2former_v1 import Mask2FormerForgeryModel
from src.utils.seed_logging_utils import setup_seed, log_seed_info

def build_train_dataset(paths, train_transform=None):
    """
    Full training dataset (same data as CV, with chosen train transform).

    Expects:
        paths: dict with keys
            - train_authentic
            - train_forged
            - train_masks
            - supp_forged (optional)
            - supp_masks (optional)
    """
    dataset = ForgeryDataset(
        paths["train_authentic"],
        paths["train_forged"],
        paths["train_masks"],
        supp_forged_path=paths.get("supp_forged"),
        supp_masks_path=paths.get("supp_masks"),
        transform=train_transform,
    )
    return dataset


def run_full_train(
    paths,
    num_epochs=30,
    batch_size=4,
    lr=1e-4,
    weight_decay=1e-4,
    device=None,
    train_transform=None,
    save_path="weights/full_train/model_full_data_baseline.pth",
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
    train_dataset = build_train_dataset(paths, train_transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Model
    model = Mask2FormerForgeryModel(
        num_queries=15,
        d_model=256,
        authenticity_penalty_weight=5.0,
        auth_gate_forged_threshold=0.5,
        default_mask_threshold=0.5,
        default_cls_threshold=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Training loop
    print("Training on FULL DATASET:", len(train_dataset), "samples")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]

            for t in targets:
                t["masks"] = t["masks"].to(device)
                t["image_label"] = t["image_label"].to(device)

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    # Save weights (to be zipped / uploaded to Kaggle later)
    torch.save(model.state_dict(), str(save_path))
    print(f"Model weights saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-data training for Mask2FormerForgeryModel "
        "(for Kaggle LB submission weights)"
    )
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

    paths = {
        "train_authentic": args.train_authentic,
        "train_forged": args.train_forged,
        "train_masks": args.train_masks,
    }
    if args.supp_forged is not None:
        paths["supp_forged"] = args.supp_forged
    if args.supp_masks is not None:
        paths["supp_masks"] = args.supp_masks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_info = setup_seed(args.seed, deterministic=args.deterministic)
    log_seed_info(seed_info)    

    run_full_train(
        paths=paths,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        train_transform=None,  # plug in the same train transform used in CV
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
