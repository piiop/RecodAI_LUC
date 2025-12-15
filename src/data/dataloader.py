# src/data/dataloader.py

"""
Dataset + DataLoader utilities for the forgery segmentation task.

Adds GroupKFold utilities to prevent leakage across authentic/forged variants
sharing the same base image id (e.g., imageA in authentic and forged).
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from sklearn.model_selection import GroupKFold  # NEW


# project_root/src/data/dataloader.py -> project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

AUTHENTIC_DIR = DATA_DIR / "train_images" / "authentic"
FORGED_DIR = DATA_DIR / "train_images" / "forged"
MASKS_DIR = DATA_DIR / "train_masks"

SUPP_FORGED_DIR = DATA_DIR / "supplemental_images"
SUPP_MASKS_DIR = DATA_DIR / "supplemental_masks"


class ForgeryDataset(Dataset):
    """
    Dataset for forged/authentic images with binary segmentation masks.
    """

    def __init__(
        self,
        transform: Optional[A.BasicTransform] = None,
        is_train: bool = True,
    ) -> None:
        self.transform = transform
        self.is_train = is_train
        self.samples: List[Dict[str, Any]] = []

        # authentic
        for file in sorted(AUTHENTIC_DIR.iterdir()):
            if not file.is_file():
                continue
            base = file.stem
            self.samples.append({
                "image_path": str(file),
                "mask_path": str(MASKS_DIR / f"{base}.npy"),
                "is_forged": False,
            })

        # forged
        for file in sorted(FORGED_DIR.iterdir()):
            if not file.is_file():
                continue
            base = file.stem
            self.samples.append({
                "image_path": str(file),
                "mask_path": str(MASKS_DIR / f"{base}.npy"),
                "is_forged": True,
            })

        # supplemental forged
        for file in sorted(SUPP_FORGED_DIR.iterdir()):
            if not file.is_file():
                continue
            base = file.stem
            self.samples.append({
                "image_path": str(file),
                "mask_path": str(SUPP_MASKS_DIR / f"{base}.npy"),
                "is_forged": True,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image_np = np.array(image)

        if os.path.exists(sample["mask_path"]):
            mask_np = np.load(sample["mask_path"])

            if mask_np.ndim == 3:
                if (
                    mask_np.shape[0] <= 15
                    and mask_np.shape[1] == image_np.shape[0]
                    and mask_np.shape[2] == image_np.shape[1]
                ):
                    mask_np = np.any(mask_np, axis=0)
                else:
                    raise ValueError(
                        f"Expected channel-first mask (C, H, W) with small C. Got {mask_np.shape}"
                    )

            mask_np = (mask_np > 0).astype(np.uint8)
        else:
            mask_np = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

        if image_np.shape[:2] != mask_np.shape:
            raise ValueError(
                f"Shape mismatch: img {image_np.shape}, mask {mask_np.shape}, file={sample['image_path']}"
            )

        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_t: torch.Tensor = transformed["image"]
            mask_t: torch.Tensor = transformed["mask"]
        else:
            image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_np.astype(np.uint8))

        if isinstance(mask_t, torch.Tensor) and mask_t.ndim == 3:
            mask_for_boxes = mask_t.squeeze(0)
        else:
            mask_for_boxes = mask_t

        if sample["is_forged"] and mask_for_boxes.sum().item() > 0:
            boxes, labels, instance_masks = self.mask_to_boxes(mask_for_boxes)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target: Dict[str, torch.Tensor] = {
                "boxes": boxes,
                "labels": labels,
                "masks": instance_masks,
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": area,
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
                "image_label": torch.tensor(1.0, dtype=torch.float32),
            }
        else:
            h, w = mask_for_boxes.shape[-2], mask_for_boxes.shape[-1]
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "image_label": torch.tensor(0.0, dtype=torch.float32),
            }

        return image_t, target

    @staticmethod
    def mask_to_boxes(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        mask_np = (mask_np > 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[List[float]] = []
        instance_masks: List[np.ndarray] = []

        for contour in contours:
            if len(contour) == 0:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 5 or h <= 5:
                continue

            boxes.append([float(x), float(y), float(x + w), float(y + h)])

            contour_mask = np.zeros_like(mask_np, dtype=np.uint8)
            cv2.fillPoly(contour_mask, [contour], 1)
            instance_masks.append(contour_mask)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones((len(boxes),), dtype=torch.int64)
            masks_t = torch.from_numpy(np.stack(instance_masks, axis=0)).to(torch.uint8)
        else:
            h, w = mask_np.shape
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)

        return boxes_t, labels_t, masks_t


def get_train_transform(img_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


# ----------------------------
# NEW: GroupKFold utilities
# ----------------------------

def get_group_ids(dataset: ForgeryDataset) -> List[str]:
    """
    Group id per sample. Using basename stem prevents leakage between
    forged/authentic variants of the same underlying image.
    """
    return [Path(s["image_path"]).stem for s in dataset.samples]


def make_groupkfold_splits(
    dataset: ForgeryDataset,
    n_splits: int = 5,
) -> List[Tuple[List[int], List[int]]]:
    """
    Returns list of (train_idx, val_idx) splits using GroupKFold.
    """
    groups = np.array(get_group_ids(dataset))
    idx = np.arange(len(dataset))

    gkf = GroupKFold(n_splits=n_splits)
    splits: List[Tuple[List[int], List[int]]] = []
    for tr, va in gkf.split(idx, y=None, groups=groups):
        splits.append((tr.tolist(), va.tolist()))
    return splits


def create_groupkfold_loaders_for_fold(
    fold: int,
    n_splits: int = 5,
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val loaders for a specific GroupKFold fold.
    """
    ds_train = ForgeryDataset(transform=get_train_transform(img_size))
    ds_val = ForgeryDataset(transform=get_val_transform(img_size))

    splits = make_groupkfold_splits(ForgeryDataset(transform=None), n_splits=n_splits)
    train_idx, val_idx = splits[fold]

    train_loader = DataLoader(
        torch.utils.data.Subset(ds_train, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        torch.utils.data.Subset(ds_val, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def create_train_val_loaders(
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: int = 256,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience helper to build train/val DataLoaders.

    Args:
        root_dir: Root directory containing train_images/, train_masks/, etc.
        batch_size: Per-GPU batch size.
        num_workers: Number of workers for the DataLoader.
        img_size: Resize side length for both train and val transforms.
        val_split: Fraction of samples to reserve for validation.
        seed: Random seed for deterministic splitting.
        use_supplemental: If True, include supplemental_images/masks if present.

    Returns:
        train_loader, val_loader
    """
    full_dataset = ForgeryDataset(
        transform=get_train_transform(img_size)
    )

    dataset_len = len(full_dataset)
    val_size = int(dataset_len * val_split)
    train_size = dataset_len - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Use validation transform for the val subset
    val_dataset.dataset.transform = get_val_transform(img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
