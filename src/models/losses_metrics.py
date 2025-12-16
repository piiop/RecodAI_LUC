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
    *,
    logger=None,
    debug_ctx: Dict | None = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Run Hungarian matching per image.

    Args:
        mask_logits: [B, Q, Hm, Wm] predicted logits
        targets: list of dicts with key 'masks': [N_gt, H, W]
        cost_bce: BCE weight for matching
        cost_dice: Dice weight for matching
        logger: optional ClsCollapseLogger-like object with .debug_event(tag, payload)
        debug_ctx: optional dict (e.g., {"epoch": 1, "global_step": 123}) to attach to logs

    Returns:
        indices: list of length B, each (pred_ind, tgt_ind) as LongTensors
    """
    B, Q, Hm, Wm = mask_logits.shape
    device = mask_logits.device
    indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
    ctx = {} if debug_ctx is None else dict(debug_ctx)

    for b in range(B):
        tgt_masks = targets[b]["masks"]  # [N_gt, H, W] or empty

        if logger is not None:
            logger.debug_event(
                "hungarian_match_input",
                {
                    **ctx,
                    "b": b,
                    "Q": int(Q),
                    "Hm": int(Hm),
                    "Wm": int(Wm),
                    "tgt_shape": list(tgt_masks.shape),
                    "tgt_numel": int(tgt_masks.numel()),
                    "tgt_sum": float(tgt_masks.sum().item()) if tgt_masks.numel() else 0.0,
                },
            )

        if tgt_masks.numel() == 0:
            empty = (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
            indices.append(empty)

            if logger is not None:
                logger.debug_event(
                    "hungarian_match_result",
                    {**ctx, "b": b, "matched": 0, "reason": "empty_gt"},
                )
            continue

        tgt_masks_resized = F.interpolate(
            tgt_masks.unsqueeze(1).float(),
            size=(Hm, Wm),
            mode="nearest",
        ).squeeze(1)  # [N_gt, Hm, Wm]

        pred = mask_logits[b]  # [Q, Hm, Wm]
        cost = match_cost(
            pred, tgt_masks_resized, cost_bce=cost_bce, cost_dice=cost_dice
        )  # [N_gt, Q]

        tgt_ind_np, pred_ind_np = linear_sum_assignment(cost.detach().cpu().numpy())
        tgt_ind = torch.as_tensor(tgt_ind_np, dtype=torch.long, device=device)
        pred_ind = torch.as_tensor(pred_ind_np, dtype=torch.long, device=device)
        indices.append((pred_ind, tgt_ind))

        if logger is not None:
            # keep it lightweight: just shapes + matched count (no cost matrix dumps)
            logger.debug_event(
                "hungarian_match_result",
                {
                    **ctx,
                    "b": b,
                    "cost_shape": [int(cost.shape[0]), int(cost.shape[1])],
                    "matched": int(pred_ind.numel()),
                    "num_gt": int(tgt_masks.shape[0]),
                    "Q": int(Q),
                },
            )

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
    loss_weight_presence_auth: float = 0.5,
    loss_weight_forged_presence: float = 0.5,
    forged_presence_tau: float = 0.10,
    presence_use_max: bool = True,
    auth_penalty_cls_threshold: float = 0.5,
    auth_penalty_temperature: float = 0.1,
    logger=None,
    debug_ctx: Dict | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute all training losses.

    - No print spam; optional structured logs via `logger.debug_event(tag, payload)`.
    - Keeps backward graph for auth penalty (no torch.no_grad).
    - Uses differentiable gating for cls in auth penalty.

    logger: optional ClsCollapseLogger-like object with .debug_event(...)
    debug_ctx: optional dict (e.g., {"epoch": 1, "global_step": 123})
    """
    B, Q, Hm, Wm = mask_logits.shape
    device = mask_logits.device
    ctx = {} if debug_ctx is None else dict(debug_ctx)

    # Hungarian matching
    indices = hungarian_match(
        mask_logits,
        targets,
        cost_bce=cost_bce,
        cost_dice=cost_dice,
        logger=logger,
        debug_ctx=debug_ctx,
    )

    # -------------------------
    # Mask losses (Dice + BCE) on matched pairs
    # -------------------------
    loss_mask_bce = mask_logits.new_zeros(())
    loss_mask_dice = mask_logits.new_zeros(())
    num_instances = 0

    for b in range(B):
        pred_ind, tgt_ind = indices[b]
        if pred_ind.numel() == 0:
            continue

        tgt_masks = targets[b]["masks"]  # [N_gt, H, W]
        tgt_masks_resized = F.interpolate(
            tgt_masks.unsqueeze(1).float(),
            size=(Hm, Wm),
            mode="nearest",
        ).squeeze(1)  # [N_gt, Hm, Wm]

        pred_masks = mask_logits[b, pred_ind]   # [M, Hm, Wm]
        gt_masks = tgt_masks_resized[tgt_ind]   # [M, Hm, Wm]

        loss_mask_bce = loss_mask_bce + sigmoid_ce_loss(pred_masks, gt_masks).sum()
        loss_mask_dice = loss_mask_dice + dice_loss(pred_masks, gt_masks).sum()
        num_instances += int(pred_ind.numel())

    if num_instances > 0:
        loss_mask_bce = loss_mask_bce / num_instances
        loss_mask_dice = loss_mask_dice / num_instances
    else:
        # keep graph/device consistent
        z = mask_logits.sum() * 0.0
        loss_mask_bce = z
        loss_mask_dice = z

    # -------------------------
    # Mask-level classification BCE:
    # matched queries -> 1, unmatched -> 0
    # -------------------------
    class_targets = torch.zeros_like(class_logits, device=device)  # [B, Q]
    for b in range(B):
        pred_ind, _ = indices[b]
        if pred_ind.numel() > 0:
            class_targets[b, pred_ind] = 1.0

    if logger is not None:
        pos = int(class_targets.sum().item())
        tot = int(class_targets.numel())
        logger.debug_event(
            "loss_cls_targets",
            {
                **ctx,
                "B": int(B),
                "Q": int(Q),
                "pos": pos,
                "total": tot,
                "pos_frac": (pos / max(tot, 1)),
            },
        )

    loss_mask_cls = F.binary_cross_entropy_with_logits(class_logits, class_targets)

    # -------------------------
    # Image-level authenticity loss
    # -------------------------
    img_targets = torch.stack([t["image_label"].float() for t in targets]).to(img_logits.device)  # [B]
    loss_img_auth = F.binary_cross_entropy_with_logits(img_logits, img_targets)

    # -------------------------
    # Differentiable authenticity penalty (ONLY on authentic images)
    # -------------------------
    mask_probs = torch.sigmoid(mask_logits)        # [B, Q, Hm, Wm]

    # -------------------------
    # Debug: mean predicted foreground prob per query (pre any inference filtering)
    # -------------------------
    if logger is not None:
        # per-query mean over batch + pixels -> [Q]
        fg_per_query = mask_probs.mean(dim=(0, 2, 3))  # [Q]
        fg_flat = fg_per_query.flatten()
        logger.debug_event(
            "train_fg_prob_per_query",
            {
                **ctx,
                "Q": int(Q),
                "fg_prob_per_query": fg_per_query.detach().cpu().tolist(),
                "fg_prob_mean": float(fg_flat.mean().item()) if fg_flat.numel() else 0.0,
                "fg_prob_p95": float(fg_flat.quantile(0.95).item()) if fg_flat.numel() else 0.0,
                "fg_prob_max": float(fg_flat.max().item()) if fg_flat.numel() else 0.0,
            },
        )
    mask_mass = mask_probs.flatten(2).mean(-1)     # [B, Q]

    # -------------------------
    # Detection presence coupling
    # presence_prob ~ "prob at least one query is positive AND has mask mass"
    # -------------------------
    cls_prob = torch.sigmoid(class_logits)         # [B, Q]
    qscore = cls_prob * mask_mass                  # [B, Q]

    if presence_use_max:
        presence_prob = qscore.max(dim=1).values   # [B]
    else:
        # smoother alternative; can be helpful if max is too peaky
        presence_prob = 1.0 - torch.exp(-qscore.sum(dim=1)).clamp(0.0, 1.0)  # [B]

    presence_prob = presence_prob.clamp(1e-6, 1.0 - 1e-6)

    # Couple presence to image label (both classes)
    loss_presence_auth = F.binary_cross_entropy(presence_prob, img_targets)

    # Enforce forged images have some detection mass (prevents "all authentic" collapse)
    forged_mask = (img_targets == 1.0).to(presence_prob.dtype)  # [B]
    tau = float(forged_presence_tau)
    # hinge: if forged and presence_prob < tau => penalty
    per_img_forged = F.relu(tau - presence_prob) * forged_mask
    # normalize by batch size (stable even if no forged in batch)
    loss_forged_presence = per_img_forged.sum() / max(B, 1)

    if logger is not None:
        logger.debug_event(
            "loss_presence_stats",
            {
                **ctx,
                "presence_mean": float(presence_prob.mean().item()) if B > 0 else 0.0,
                "presence_min": float(presence_prob.min().item()) if B > 0 else 0.0,
                "presence_max": float(presence_prob.max().item()) if B > 0 else 0.0,
                "tau": tau,
                "loss_presence_auth": float(loss_presence_auth.detach().cpu()),
                "loss_forged_presence": float(loss_forged_presence.detach().cpu()),
            },
        )

    thr = float(auth_penalty_cls_threshold)
    temp = max(float(auth_penalty_temperature), 1e-6)
    soft_cls_weight = torch.sigmoid((class_logits - thr) / temp)  # [B, Q]

    per_image_penalty = (soft_cls_weight * mask_mass).mean(dim=1)  # [B]
    authentic_mask = (img_targets == 0.0).to(per_image_penalty.dtype)  # [B]
    penalty = (per_image_penalty * authentic_mask).sum() / max(B, 1)
    loss_auth_penalty = authenticity_penalty_weight * penalty

    if logger is not None:
        logger.debug_event(
            "loss_auth_penalty_stats",
            {
                **ctx,
                "thr": thr,
                "temp": temp,
                "authentic_frac": float(authentic_mask.mean().item()) if B > 0 else 0.0,
                "per_image_penalty_mean": float(per_image_penalty.mean().item()) if B > 0 else 0.0,
                "loss_auth_penalty": float(loss_auth_penalty.detach().cpu()),
            },
        )

    # -------------------------
    # Weighted total loss
    # -------------------------
    loss_total = (
        loss_weight_mask_bce * loss_mask_bce
        + loss_weight_mask_dice * loss_mask_dice
        + loss_weight_mask_cls * loss_mask_cls
        + loss_weight_img_auth * loss_img_auth
        + loss_weight_auth_penalty * loss_auth_penalty
        + loss_weight_presence_auth * loss_presence_auth
        + loss_weight_forged_presence * loss_forged_presence        
    )

    return {
        "loss_mask_bce": loss_mask_bce,
        "loss_mask_dice": loss_mask_dice,
        "loss_mask_cls": loss_mask_cls,
        "loss_img_auth": loss_img_auth,
        "loss_auth_penalty": loss_auth_penalty,
        "loss_presence_auth": loss_presence_auth,
        "loss_forged_presence": loss_forged_presence,        
        "loss_total": loss_total,
    }