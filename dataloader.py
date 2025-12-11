import os
import cv2
import json
import sympy
import sympy.printing  # ensure submodule is imported
sympy.printing = sympy.printing  # attach as attribute explicitly
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.ops import FeaturePyramidNetwork

from scipy.optimize import linear_sum_assignment

import albumentations as A
import matplotlib.pyplot as plt


from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from sklearn.model_selection import train_test_split
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F_transforms

def analyze_data_structure():
    base_path = '.'
    
    # Checking train images
    train_authentic_path = os.path.join(base_path, 'train_images/authentic')
    train_forged_path = os.path.join(base_path, 'train_images/forged')
    train_masks_path = os.path.join(base_path, 'train_masks')
    test_images_path = os.path.join(base_path, 'test_images')
    supp_forged_path = os.path.join(base_path, 'supplemental_images')
    supp_masks_path = os.path.join(base_path, 'supplemental_masks')
    
    print(f"Authentic images: {len(os.listdir(train_authentic_path))}")
    print(f"Forged images: {len(os.listdir(train_forged_path))}")
    print(f"Masks: {len(os.listdir(train_masks_path))}")
    print(f"Test images: {len(os.listdir(test_images_path))}")

    print(f"Supp forged images: {len(os.listdir(supp_forged_path))}")
    print(f"Supp masks: {len(os.listdir(supp_masks_path))}")
    
    # Let's analyze some examples of masks
    mask_files = os.listdir(train_masks_path)[:5]
    print(f"Examples of mask files: {mask_files}")
    
    # Checking the mask format
    sample_mask = np.load(os.path.join(train_masks_path, mask_files[0]))
    print(f"Mask format: {sample_mask.shape}, dtype: {sample_mask.dtype}")
    
    test_files = os.listdir(test_images_path)
    print(f"Test images: {test_files}")
    
    return {
        'train_authentic': train_authentic_path,
        'train_forged': train_forged_path,
        'train_masks': train_masks_path,
        'test_images': test_images_path,
        'supp_forged': supp_forged_path,
        'supp_masks' : supp_masks_path
    }

paths = analyze_data_structure()

class ForgeryDataset(Dataset):
    def __init__(
        self,
        authentic_path,
        forged_path,
        masks_path,
        supp_forged_path=None,
        supp_masks_path=None,
        transform=None,
        is_train=True,
    ):
        self.transform = transform
        self.is_train = is_train
        
        # Collect all data samples
        self.samples = []
        
        # Authentic images
        for file in os.listdir(authentic_path):
            img_path = os.path.join(authentic_path, file)
            base_name = file.split('.')[0]
            mask_path = os.path.join(masks_path, f"{base_name}.npy")
            
            self.samples.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'is_forged': False,
                'image_id': base_name
            })
        
        # Forged images (original)
        for file in os.listdir(forged_path):
            img_path = os.path.join(forged_path, file)
            base_name = file.split('.')[0]
            mask_path = os.path.join(masks_path, f"{base_name}.npy")
            
            self.samples.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'is_forged': True,
                'image_id': base_name
            })

        # Supplemental forged images (all forged)
        if supp_forged_path is not None and supp_masks_path is not None:
            for file in os.listdir(supp_forged_path):
                img_path = os.path.join(supp_forged_path, file)
                base_name = file.split('.')[0]
                mask_path = os.path.join(supp_masks_path, f"{base_name}.npy")
                
                self.samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'is_forged': True,
                    'image_id': base_name
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)  # (H, W, 3)
        
        # Load and process mask
        if os.path.exists(sample['mask_path']):
            mask = np.load(sample['mask_path'])
            
            # Handle multi-channel masks: MUST be channel-first (C, H, W)
            if mask.ndim == 3:
                # Expect small channel dimension as first dim
                if mask.shape[0] <= 15 and mask.shape[1:] == image.shape[:2]:
                    mask = np.any(mask, axis=0)
                else:
                    raise ValueError(
                        f"Expected channel-first mask (C, H, W) with small C. Got {mask.shape}"
                    )
            
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
        # Shape validation
        assert image.shape[:2] == mask.shape, f"Shape mismatch: img {image.shape}, mask {mask.shape}"
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = F_transforms.to_tensor(image)
            mask = torch.tensor(mask, dtype=torch.uint8)
        
        # Prepare targets for Mask2Former
        if sample['is_forged'] and mask.sum() > 0:
            boxes, labels, masks = self.mask_to_boxes(mask)
            
            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
            }
            target['image_label'] = torch.tensor(1.0)   # forged
        else:
            # For authentic images or images without masks
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'masks': torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
            target['image_label'] = torch.tensor(0.0)   # authentic
        return image, target
    
    def mask_to_boxes(self, mask):
        """Convert segmentation mask to bounding boxes for Mask2Former"""
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        masks = []
        
        for contour in contours:
            if len(contour) > 0:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out very small regions
                if w > 5 and h > 5:
                    boxes.append([x, y, x + w, y + h])
                    # Create binary mask for this contour
                    contour_mask = np.zeros_like(mask_np)
                    cv2.fillPoly(contour_mask, [contour], 1)
                    masks.append(contour_mask)
        
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, mask_np.shape[0], mask_np.shape[1]), dtype=torch.uint8)
        
        return boxes, labels, masks
    
    # Transformations for learning, ImageNet standards
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

full_dataset = ForgeryDataset(
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
    supp_forged_path=paths['supp_forged'],
    supp_masks_path=paths['supp_masks'],
    transform=train_transform
)

# Split into train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Use val transforms for validation subset
val_dataset.dataset.transform = val_transform

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
