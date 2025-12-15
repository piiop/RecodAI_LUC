"""
Losses and matching utilities for Mask2Former-style forgery model.

Provides:
- Hungarian matching with BCE + Dice cost
- Per-instance BCE and soft Dice losses
- Full loss computation (mask, class, image-level auth, auth penalty)
"""

from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ------------------- Basic losses -------------------


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    BCE on logits, per-instance mean.

    Args:
        inputs: [M, H, W] logits
        targets: [M, H, W] binary targets

    Returns:
        loss: [M] per-instance loss
    """
    return F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    ).mean(dim=(1, 2))


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft dice loss on logits.

    Args:
        inputs: [M, H, W] logits
        targets: [M, H, W] binary targets

    Returns:
        loss: [M] per-instance loss
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1) + eps
    loss = 1.0 - (numerator + eps) / (denominator)
    return loss


# ------------------- Matching cost & Hungarian matching -------------------


def match_cost(
    pred_masks: torch.Tensor,
    tgt_masks: torch.Tensor,
    cost_bce: float = 1.0,
    cost_dice: float = 1.0,
) -> torch.Tensor:
    """
    Compute match cost matrix between predicted and GT masks.

    Args:
        pred_masks: [Q, H, W] predicted logits
        tgt_masks: [N_gt, H, W] GT binary masks
        cost_bce: weight for BCE term
        cost_dice: weight for Dice term

    Returns:
        cost: [N_gt, Q] cost matrix
    """
    Q, H, W = pred_masks.shape
    N = tgt_masks.shape[0]

    pred_flat = pred_masks.flatten(1)  # [Q, HW]
    tgt_flat = tgt_masks.flatten(1)    # [N, HW]

    # BCE cost
    pred_logits = pred_flat.unsqueeze(0)  # [1, Q, HW]
    tgt = tgt_flat.unsqueeze(1)           # [N, 1, HW]
    bce = F.binary_cross_entropy_with_logits(
        pred_logits.expand(N, -1, -1),
        tgt.expand(-1, Q, -1),
        reduction="none",
    ).mean(-1)  # [N, Q]

    # Dice cost
    pred_prob = pred_flat.sigmoid()
    numerator = 2 * (pred_prob.unsqueeze(0) * tgt_flat.unsqueeze(1)).sum(-1)
    denominator = pred_prob.unsqueeze(0).sum(-1) + tgt_flat.unsqueeze(1).sum(-1) + 1e-6
    dice = 1.0 - (numerator + 1e-6) / (denominator)

    cost = cost_bce * bce + cost_dice * dice
    return cost


def hungarian_match(
    mask_logits: torch.Tensor,
    targets: List[Dict],
    cost_bce: float = 1.0,
    cost_dice: float = 1.0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Run Hungarian matching per image.

    Args:
        mask_logits: [B, Q, Hm, Wm] predicted logits
        targets: list of dicts with key 'masks': [N_gt, H, W]
        cost_bce: BCE weight for matching
        cost_dice: Dice weight for matching

    Returns:
        indices: list of length B, each (pred_ind, tgt_ind) as LongTensors
    """
    B, Q, Hm, Wm = mask_logits.shape
    device = mask_logits.device
    indices: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for b in range(B):
        tgt_masks = targets[b]["masks"]  # [N_gt, H, W] or empty
        print("[MATCH] b", b,
              "tgt_shape", tuple(tgt_masks.shape),
              "tgt_numel", int(tgt_masks.numel()),
              "tgt_sum", float(tgt_masks.sum().item()))
        if tgt_masks.numel() == 0:
            empty = (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
            indices.append(empty)
            continue

        tgt_masks_resized = F.interpolate(
            tgt_masks.unsqueeze(1).float(),
            size=(Hm, Wm),
            mode="nearest",
        ).squeeze(1)  # [N_gt, Hm, Wm]

        pred = mask_logits[b]  # [Q, Hm, Wm]
        cost = match_cost(pred, tgt_masks_resized, cost_bce=cost_bce, cost_dice=cost_dice)  # [N_gt, Q]

        tgt_ind, pred_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        print("[MATCH] cost_shape", tuple(cost.shape),
                "matched", len(pred_ind), "Q", Q)
        tgt_ind = torch.as_tensor(tgt_ind, dtype=torch.long, device=device)
        pred_ind = torch.as_tensor(pred_ind, dtype=torch.long, device=device)
        indices.append((pred_ind, tgt_ind))

    return indices


# ------------------- Full loss computation -------------------

def compute_losses(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    img_logits: torch.Tensor,
    targets: List[Dict],
    *,
    cost_bce: float = 1.0,
    cost_dice: float = 1.0,
    loss_weight_mask_bce: float = 1.0,
    loss_weight_mask_dice: float = 1.0,
    loss_weight_mask_cls: float = 1.0,
    loss_weight_img_auth: float = 1.0,
    loss_weight_auth_penalty: float = 1.0,
    authenticity_penalty_weight: float = 5.0,
    auth_penalty_cls_threshold: float = 0.5,
    auth_penalty_temperature: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Compute all training losses.

    Fixes:
      - removes torch.no_grad() around auth penalty
      - replaces hard cls threshold with differentiable soft weighting
    """
    B, Q, Hm, Wm = mask_logits.shape
    device = mask_logits.device

    # Hungarian matching
    indices = hungarian_match(
        mask_logits,
        targets,
        cost_bce=cost_bce,
        cost_dice=cost_dice,
    )

    # -------------------------
    # Mask losses (Dice + BCE) on matched pairs
    # -------------------------
    loss_mask = mask_logits.new_zeros(())
    loss_dice_val = mask_logits.new_zeros(())
    num_instances = 0

    for b in range(B):
        pred_ind, tgt_ind = indices[b]
        if len(pred_ind) == 0:
            continue

        tgt_masks = targets[b]["masks"]  # [N_gt, H, W]
        tgt_masks_resized = F.interpolate(
            tgt_masks.unsqueeze(1).float(),
            size=(Hm, Wm),
            mode="nearest",
        ).squeeze(1)  # [N_gt, Hm, Wm]

        pred_masks = mask_logits[b, pred_ind]         # [M, Hm, Wm]
        gt_masks = tgt_masks_resized[tgt_ind]         # [M, Hm, Wm]

        loss_mask = loss_mask + sigmoid_ce_loss(pred_masks, gt_masks).sum()
        loss_dice_val = loss_dice_val + dice_loss(pred_masks, gt_masks).sum()
        num_instances += pred_ind.numel()

    if num_instances > 0:
        loss_mask = loss_mask / num_instances
        loss_dice_val = loss_dice_val / num_instances
    else:
        # keep graph / device
        zero = mask_logits.sum() * 0.0
        loss_mask = zero
        loss_dice_val = zero

    # -------------------------
    # Mask-level classification BCE
    # matched queries -> 1, unmatched -> 0
    # -------------------------
    class_targets = torch.zeros_like(class_logits, device=device)  # [B, Q]
    for b in range(B):
        pred_ind, _ = indices[b]
        if len(pred_ind) > 0:
            class_targets[b, pred_ind] = 1.0

    loss_cls = F.binary_cross_entropy_with_logits(class_logits, class_targets)

    # -------------------------
    # Image-level authenticity loss
    # -------------------------
    img_targets = torch.stack([t["image_label"].float() for t in targets]).to(img_logits.device)  # [B]
    loss_img = F.binary_cross_entropy_with_logits(img_logits, img_targets)

    # -------------------------
    # Differentiable authenticity penalty (ONLY on authentic images)
    #
    # Penalize "expected forged mask mass" on authentic images:
    #   soft_cls_weight_q = sigmoid((cls_logit_q - thr) / temp)
    #   mask_mass_q = mean(sigmoid(mask_logit_q)) over pixels
    #   penalty_b = mean_q (soft_cls_weight_q * mask_mass_q)
    # -------------------------
    mask_probs = torch.sigmoid(mask_logits)     # [B, Q, Hm, Wm]
    mask_mass = mask_probs.flatten(2).mean(-1)  # [B, Q]

    thr = float(auth_penalty_cls_threshold)
    temp = max(float(auth_penalty_temperature), 1e-6)
    soft_cls_weight = torch.sigmoid((class_logits - thr) / temp)  # [B, Q]

    per_image_penalty = (soft_cls_weight * mask_mass).mean(dim=1)  # [B]

    # apply only where img_targets == 0 (authentic)
    authentic_mask = (img_targets == 0.0).to(per_image_penalty.dtype)  # [B]
    penalty = (per_image_penalty * authentic_mask).sum() / max(B, 1)

    loss_auth_penalty = authenticity_penalty_weight * penalty

    # -------------------------
    # Weighted total loss
    # -------------------------
    loss_total = (
        loss_weight_mask_bce * loss_mask +
        loss_weight_mask_dice * loss_dice_val +
        loss_weight_mask_cls * loss_cls +
        loss_weight_img_auth * loss_img +
        loss_weight_auth_penalty * loss_auth_penalty
    )

    return {
        "loss_mask_bce": loss_mask,
        "loss_mask_dice": loss_dice_val,
        "loss_mask_cls": loss_cls,
        "loss_img_auth": loss_img,
        "loss_auth_penalty": loss_auth_penalty,
        "loss_total": loss_total,
    }

