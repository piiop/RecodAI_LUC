"""
Dataset and dataloader utilities for the forgery segmentation task.

Expected directory layout (per run/config):

    root/
    ├── train_images/
    │   ├── authentic/          # authentic RGB images
    │   └── forged/             # forged  RGB images
    ├── train_masks/            # .npy masks, one per image (same basename)
    ├── supplemental_images/    # (optional) extra forged images
    └── supplemental_masks/     # (optional) extra masks for supplemental_images

Each image has a corresponding mask file:
    <basename>.jpg/.png/...  ->  train_masks/<basename>.npy

Minimal usage (inside your training script):

    from src.data.dataloader import (
        ForgeryDataset,
        get_train_transform,
        get_val_transform,
        create_train_val_loaders,
    )

    train_loader, val_loader = create_train_val_loaders(
        root_dir="/path/to/dataset/root",
        batch_size=4,
        num_workers=4,
        img_size=256,
        val_split=0.2,
        seed=42,
        use_supplemental=True,
    )

    for images, targets in train_loader:
        # images: list[Tensor(C,H,W)], already normalized
        # targets: list[dict] with boxes, labels, masks, image_label, ...
        ...

This module is CPU-side only; CUDA usage is handled in the training loop by
sending `images` and each `target` dict to your device.
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


class ForgeryDataset(Dataset):
    """
    Dataset for forged/authentic images with binary segmentation masks.

    Args:
        authentic_path: Directory with authentic images.
        forged_path: Directory with forged images.
        masks_path: Directory with .npy masks (basenames matching image files).
        supp_forged_path: Optional directory with supplemental forged images.
        supp_masks_path: Optional directory with masks for supplemental images.
        transform: Albumentations transform applied to (image, mask).
        is_train: Unused flag kept for compatibility / future use.

    Returns:
        __getitem__ -> (image, target) where:
            image: Tensor of shape (3, H, W), normalized if transform includes it.
            target: dict with keys:
                - boxes: FloatTensor [N, 4] (x_min, y_min, x_max, y_max)
                - labels: LongTensor [N] (1 for forged objects)
                - masks: UInt8Tensor [N, H, W]
                - image_id: LongTensor [1]
                - area: FloatTensor [N]
                - iscrowd: LongTensor [N]
                - image_label: FloatTensor scalar (1.0 forged, 0.0 authentic)
    """

    def __init__(
        self,
        authentic_path: str,
        forged_path: str,
        masks_path: str,
        supp_forged_path: Optional[str] = None,
        supp_masks_path: Optional[str] = None,
        transform: Optional[A.BasicTransform] = None,
        is_train: bool = True,
    ) -> None:
        self.transform = transform
        self.is_train = is_train  # reserved for future behavior toggles

        self.samples: List[Dict[str, Any]] = []

        # Authentic images (sorted for reproducibility)
        if os.path.isdir(authentic_path):
            for file in sorted(os.listdir(authentic_path)):
                img_path = os.path.join(authentic_path, file)
                if not os.path.isfile(img_path):
                    continue

                base_name, _ = os.path.splitext(file)
                mask_path = os.path.join(masks_path, f"{base_name}.npy")

                self.samples.append(
                    {
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "is_forged": False,
                        "image_id": base_name,
                    }
                )

        # Forged images (original)
        if os.path.isdir(forged_path):
            for file in sorted(os.listdir(forged_path)):
                img_path = os.path.join(forged_path, file)
                if not os.path.isfile(img_path):
                    continue

                base_name, _ = os.path.splitext(file)
                mask_path = os.path.join(masks_path, f"{base_name}.npy")

                self.samples.append(
                    {
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "is_forged": True,
                        "image_id": base_name,
                    }
                )

        # Supplemental forged images (all forged)
        if supp_forged_path is not None and supp_masks_path is not None:
            if os.path.isdir(supp_forged_path):
                for file in sorted(os.listdir(supp_forged_path)):
                    img_path = os.path.join(supp_forged_path, file)
                    if not os.path.isfile(img_path):
                        continue

                    base_name, _ = os.path.splitext(file)
                    mask_path = os.path.join(supp_masks_path, f"{base_name}.npy")

                    self.samples.append(
                        {
                            "image_path": img_path,
                            "mask_path": mask_path,
                            "is_forged": True,
                            "image_id": base_name,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]

        # Load image (H, W, 3)
        image = Image.open(sample["image_path"]).convert("RGB")
        image_np = np.array(image)

        # Load and process mask
        if os.path.exists(sample["mask_path"]):
            mask_np = np.load(sample["mask_path"])

            # Handle multi-channel masks: expect (C, H, W) with small C
            if mask_np.ndim == 3:
                # channel-first with small C and spatial dims matching image
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
            mask_np = np.zeros(
                (image_np.shape[0], image_np.shape[1]), dtype=np.uint8
            )

        # Shape validation
        if image_np.shape[:2] != mask_np.shape:
            raise ValueError(
                f"Shape mismatch: img {image_np.shape}, mask {mask_np.shape}, file={sample['image_path']}"
            )

        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_t: torch.Tensor = transformed["image"]
            mask_t: torch.Tensor = transformed["mask"]
        else:
            # Fallback: simple tensor conversion, no normalization
            image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_np.astype(np.uint8))

        # Ensure mask is 2D for contour extraction
        if isinstance(mask_t, torch.Tensor) and mask_t.ndim == 3:
            # e.g. (1, H, W) -> (H, W)
            mask_for_boxes = mask_t.squeeze(0)
        else:
            mask_for_boxes = mask_t

        # Prepare targets
        if sample["is_forged"] and mask_for_boxes.sum().item() > 0:
            boxes, labels, instance_masks = self.mask_to_boxes(mask_for_boxes)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target: Dict[str, torch.Tensor] = {
                "boxes": boxes,
                "labels": labels,
                "masks": instance_masks,
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": area,
                "iscrowd": torch.zeros(
                    (len(boxes),), dtype=torch.int64
                ),
                "image_label": torch.tensor(1.0, dtype=torch.float32),
            }
        else:
            # Authentic images or images without positive mask
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
        """
        Convert a binary segmentation mask to bounding boxes and instance masks.

        Args:
            mask: Tensor or ndarray of shape (H, W) with {0,1}.

        Returns:
            boxes: FloatTensor [N, 4] in (x_min, y_min, x_max, y_max) format.
            labels: LongTensor [N], all ones (class id 1).
            masks: UInt8Tensor [N, H, W], instance masks per box.
        """
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        # Ensure uint8, contiguous
        mask_np = (mask_np > 0).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: List[List[float]] = []
        instance_masks: List[np.ndarray] = []

        for contour in contours:
            if len(contour) == 0:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Filter out tiny regions that are likely noise
            if w <= 5 or h <= 5:
                continue

            boxes.append([float(x), float(y), float(x + w), float(y + h)])

            contour_mask = np.zeros_like(mask_np, dtype=np.uint8)
            cv2.fillPoly(contour_mask, [contour], 1)
            instance_masks.append(contour_mask)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones((len(boxes),), dtype=torch.int64)
            masks_t = torch.from_numpy(np.stack(instance_masks, axis=0)).to(
                torch.uint8
            )
        else:
            h, w = mask_np.shape
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)

        return boxes_t, labels_t, masks_t


def get_train_transform(img_size: int = 256) -> A.Compose:
    """Augmentation pipeline for training."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size: int = 256) -> A.Compose:
    """Evaluation/validation transform (no heavy augmentation)."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for detection/segmentation models.

    Keeps images and targets as lists (variable number of instances per image).
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_train_val_loaders(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: int = 256,
    val_split: float = 0.2,
    seed: int = 42,
    use_supplemental: bool = True,
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
    authentic_path = os.path.join(root_dir, "train_images", "authentic")
    forged_path = os.path.join(root_dir, "train_images", "forged")
    masks_path = os.path.join(root_dir, "train_masks")

    supp_forged_path = (
        os.path.join(root_dir, "supplemental_images") if use_supplemental else None
    )
    supp_masks_path = (
        os.path.join(root_dir, "supplemental_masks") if use_supplemental else None
    )

    full_dataset = ForgeryDataset(
        authentic_path=authentic_path,
        forged_path=forged_path,
        masks_path=masks_path,
        supp_forged_path=supp_forged_path,
        supp_masks_path=supp_masks_path,
        transform=get_train_transform(img_size),
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
