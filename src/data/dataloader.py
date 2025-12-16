# src/data/dataloader.py

"""
Dataset + DataLoader utilities for the forgery segmentation task.

REFRACTOR (Decouple masks from class truth):
- Class truth (image_label) comes ONLY from folder:
    train_images/authentic -> 0
    train_images/forged    -> 1
    supplemental_images    -> 1
- Masks are localization only:
    - For forged images: use mask instances if present (K,H,W or H,W)
    - For authentic images: ALWAYS return empty GT instances (even if mask exists)
      (authentic masks are "where forgery would be", not a positive label)
- GroupKFold groups by image stem to prevent leakage across authentic/forged variants.

This file is used by train_cv via make_groupkfold_splits().  See train_cv.py usage. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset


# project_root/src/data/dataloader.py -> project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

AUTHENTIC_DIR = DATA_DIR / "train_images" / "authentic"
FORGED_DIR = DATA_DIR / "train_images" / "forged"
MASKS_DIR = DATA_DIR / "train_masks"

SUPP_FORGED_DIR = DATA_DIR / "supplemental_images"
SUPP_MASKS_DIR = DATA_DIR / "supplemental_masks"


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_mask_instances(mask_path: Path, *, expected_h: int, expected_w: int) -> np.ndarray:
    """
    Load mask as instance stack [N, H, W] uint8 in {0,1}.

    Supports:
      - (H,W) single mask  -> treated as 1 instance if any fg
      - (K,H,W) instances  -> each channel is its own instance
    """
    if not mask_path.exists():
        return np.zeros((0, expected_h, expected_w), dtype=np.uint8)

    m = np.load(str(mask_path))

    if m.ndim == 2:
        if m.shape != (expected_h, expected_w):
            raise ValueError(f"Mask shape mismatch: got {m.shape}, expected {(expected_h, expected_w)} for {mask_path}")
        m = (m > 0).astype(np.uint8)
        if m.sum() == 0:
            return np.zeros((0, expected_h, expected_w), dtype=np.uint8)
        return m[None, :, :]

    if m.ndim == 3:
        # Expect (K,H,W)
        if m.shape[1] != expected_h or m.shape[2] != expected_w:
            raise ValueError(
                f"Mask shape mismatch: got {m.shape}, expected (K,{expected_h},{expected_w}) for {mask_path}"
            )
        m = (m > 0).astype(np.uint8)

        # Drop empty channels (common if K is fixed)
        keep = m.reshape(m.shape[0], -1).sum(axis=1) > 0
        m = m[keep]
        if m.shape[0] == 0:
            return np.zeros((0, expected_h, expected_w), dtype=np.uint8)
        return m

    raise ValueError(f"Unsupported mask ndim={m.ndim} for {mask_path}")


def _instances_to_boxes(instances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    instances: [N,H,W] uint8
    returns:
      boxes: [N,4] float32 in xyxy
      areas: [N] float32
    """
    if instances.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes: List[List[float]] = []
    areas: List[float] = []
    for i in range(instances.shape[0]):
        mask = instances[i]
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        # +1 to make box inclusive-ish; matches typical mask->box convention
        boxes.append([float(x0), float(y0), float(x1 + 1), float(y1 + 1)])
        areas.append(float((x1 - x0 + 1) * (y1 - y0 + 1)))

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.asarray(boxes, dtype=np.float32), np.asarray(areas, dtype=np.float32)


class ForgeryDataset(Dataset):
    """
    Dataset for forged/authentic images with *instance* masks.

    Key rule:
      - image_label is determined ONLY by directory (authentic=0, forged=1).
      - masks are localization only; authentic samples always have empty instance GT.
    """

    def __init__(
        self,
        transform: Optional[A.BasicTransform] = None,
        is_train: bool = True,
    ) -> None:
        self.transform = transform
        self.is_train = is_train
        self.samples: List[Dict[str, Any]] = []

        # authentic (label=0)  — mask path exists but MUST NOT create positive GT instances
        if AUTHENTIC_DIR.exists():
            for file in sorted(AUTHENTIC_DIR.iterdir()):
                if not _is_image_file(file):
                    continue
                base = file.stem
                self.samples.append(
                    {
                        "image_path": str(file),
                        "mask_path": str(MASKS_DIR / f"{base}.npy"),
                        "is_forged": False,
                    }
                )

        # forged (label=1)
        if FORGED_DIR.exists():
            for file in sorted(FORGED_DIR.iterdir()):
                if not _is_image_file(file):
                    continue
                base = file.stem
                self.samples.append(
                    {
                        "image_path": str(file),
                        "mask_path": str(MASKS_DIR / f"{base}.npy"),
                        "is_forged": True,
                    }
                )

        # supplemental forged (label=1)
        if SUPP_FORGED_DIR.exists():
            for file in sorted(SUPP_FORGED_DIR.iterdir()):
                if not _is_image_file(file):
                    continue
                base = file.stem
                self.samples.append(
                    {
                        "image_path": str(file),
                        "mask_path": str(SUPP_MASKS_DIR / f"{base}.npy"),
                        "is_forged": True,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image_np = np.array(image)
        h0, w0 = image_np.shape[0], image_np.shape[1]

        is_forged: bool = bool(sample["is_forged"])
        image_label = 1.0 if is_forged else 0.0

        # ---- Localization GT (decoupled) ----
        # Authentic: always empty GT instances (even if mask exists)
        if not is_forged:
            inst_np = np.zeros((0, h0, w0), dtype=np.uint8)
            # still pass a dummy 2D mask through albumentations so transforms stay consistent
            union_mask_np = np.zeros((h0, w0), dtype=np.uint8)
        else:
            inst_np = _load_mask_instances(Path(sample["mask_path"]), expected_h=h0, expected_w=w0)
            # union only used for albumentations mask transform routing
            union_mask_np = (inst_np.any(axis=0).astype(np.uint8) if inst_np.shape[0] > 0 else np.zeros((h0, w0), np.uint8))

        # ---- Transforms ----
        if self.transform is not None:
            # Albumentations supports masks as HxW; we transform union, then reproject instances by re-running transform per instance
            # (cheap because K is small; preserves instance structure after resize/flip)
            transformed = self.transform(image=image_np, mask=union_mask_np)
            image_t: torch.Tensor = transformed["image"]
            union_t: torch.Tensor = transformed["mask"]

            # If forged, transform each instance with same replay to keep identical aug
            if is_forged and inst_np.shape[0] > 0:
                # Use ReplayCompose if available; otherwise fall back to union-only instances (connected components)
                if isinstance(self.transform, A.ReplayCompose):
                    rep = self.transform(image=image_np, mask=union_mask_np)
                    image_t = rep["image"]
                    inst_list = []
                    for k in range(inst_np.shape[0]):
                        mk = A.ReplayCompose.replay(rep["replay"], image=image_np, mask=inst_np[k])["mask"]
                        inst_list.append(mk)
                    inst_t = torch.stack(inst_list, dim=0).to(torch.uint8)
                else:
                    # No deterministic replay => safest is to derive instances after transform from union
                    # (still valid supervision, just loses per-channel instance separation)
                    ut = union_t
                    if isinstance(ut, torch.Tensor):
                        ut_np = ut.detach().cpu().numpy().astype(np.uint8)
                    else:
                        ut_np = np.asarray(ut, dtype=np.uint8)
                    inst_cc = self._union_to_connected_components(ut_np)
                    inst_t = torch.from_numpy(inst_cc).to(torch.uint8)
            else:
                # authentic => empty instances at transformed resolution
                ht, wt = int(union_t.shape[-2]), int(union_t.shape[-1])
                inst_t = torch.zeros((0, ht, wt), dtype=torch.uint8)
        else:
            image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            inst_t = torch.from_numpy(inst_np).to(torch.uint8)

        # ---- Targets ----
        if inst_t.numel() == 0:
            ht, wt = int(image_t.shape[-2]), int(image_t.shape[-1])
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, ht, wt), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "image_label": torch.tensor(image_label, dtype=torch.float32),
            }
            return image_t, target

        # boxes/areas from transformed instances
        inst_np2 = inst_t.detach().cpu().numpy().astype(np.uint8)
        boxes_np, areas_np = _instances_to_boxes(inst_np2)

        target = {
            "boxes": torch.from_numpy(boxes_np).to(torch.float32),
            "labels": torch.ones((boxes_np.shape[0],), dtype=torch.int64),
            "masks": inst_t.to(torch.uint8),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.from_numpy(areas_np).to(torch.float32),
            "iscrowd": torch.zeros((boxes_np.shape[0],), dtype=torch.int64),
            "image_label": torch.tensor(image_label, dtype=torch.float32),
        }
        return image_t, target

    @staticmethod
    def _union_to_connected_components(union_mask_hw: np.ndarray) -> np.ndarray:
        """
        Fallback instance extraction from a union mask after transform (used when not ReplayCompose).
        Returns [N,H,W] uint8.
        """
        m = (union_mask_hw > 0).astype(np.uint8)
        if m.sum() == 0:
            return np.zeros((0, m.shape[0], m.shape[1]), dtype=np.uint8)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inst: List[np.ndarray] = []
        for c in contours:
            if len(c) == 0:
                continue
            cc = np.zeros_like(m, dtype=np.uint8)
            cv2.fillPoly(cc, [c], 1)
            inst.append(cc)
        if not inst:
            return np.zeros((0, m.shape[0], m.shape[1]), dtype=np.uint8)
        return np.stack(inst, axis=0).astype(np.uint8)


def get_train_transform(img_size: int = 256) -> A.BasicTransform:
    # NOTE: Using ReplayCompose enables consistent per-instance transforms.
    return A.ReplayCompose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size: int = 256) -> A.BasicTransform:
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
# GroupKFold utilities
# ----------------------------

def get_group_ids(dataset: ForgeryDataset) -> List[str]:
    """
    Group id per sample: the basename stem.
    This ensures authentic/forged variants of the same ID never split across folds.
    """
    return [Path(s["image_path"]).stem for s in dataset.samples]


def make_groupkfold_splits(
    dataset: ForgeryDataset,
    n_splits: int = 5,
) -> List[Tuple[List[int], List[int]]]:
    """
    Returns list of (train_idx, val_idx) using GroupKFold by image stem.
    """
    groups = np.asarray(get_group_ids(dataset))
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
    # use ReplayCompose for train so instances stay aligned post-aug
    ds_train = ForgeryDataset(transform=get_train_transform(img_size))
    ds_val = ForgeryDataset(transform=get_val_transform(img_size))

    base_ds = ForgeryDataset(transform=None)
    splits = make_groupkfold_splits(base_ds, n_splits=n_splits)
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
