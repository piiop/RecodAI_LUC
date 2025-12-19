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
    targets: List[Dict],
    *,
    cost_bce: float = 1.0,
    cost_dice: float = 1.0,
    # (initially) treat all as 1.0; keep args for config compatibility
    loss_weight_mask_bce: float = 1.0,
    loss_weight_mask_dice: float = 1.0,
    loss_weight_mask_cls: float = 1.0,
    loss_weight_presence: float = 1.0,
    loss_weight_auth_penalty: float = 1.0,
    authenticity_penalty_weight: float = 1.0,
    logger=None,
    debug_ctx: Dict | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Refactored losses (no image head supervision; no thresholds/temperature in losses).

    Keeps:
      1) Instance mask loss on matched pairs (Dice + BCE)
      2) Query classification BCE (matched=1, unmatched=0)
      3) Presence BCE vs image_label using a differentiable presence_prob

    Suppression:
      - loss_auth_penalty: penalize detection "mass" on authentic images (no thresholds),
        implemented as mean over queries of (cls_prob * mask_mass).

    Notes:
      - img_logits is unused (image head is to be removed elsewhere).
      - presence_prob is differentiable and uses max over queries (no extra knobs).
    """
    B, Q, Hm, Wm = mask_logits.shape
    device = mask_logits.device
    ctx = {} if debug_ctx is None else dict(debug_ctx)

    # -------------------------
    # Hungarian matching
    # -------------------------
    indices = hungarian_match(
        mask_logits,
        targets,
        cost_bce=cost_bce,
        cost_dice=cost_dice,
        logger=logger,
        debug_ctx=debug_ctx,
    )

    # -------------------------
    # (1) Mask losses (Dice + BCE) on matched pairs
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

        pred_masks = mask_logits[b, pred_ind]        # [M, Hm, Wm]
        gt_masks = tgt_masks_resized[tgt_ind]        # [M, Hm, Wm]

        loss_mask_bce = loss_mask_bce + sigmoid_ce_loss(pred_masks, gt_masks).sum()
        loss_mask_dice = loss_mask_dice + dice_loss(pred_masks, gt_masks).sum()
        num_instances += int(pred_ind.numel())

    if num_instances > 0:
        loss_mask_bce = loss_mask_bce / num_instances
        loss_mask_dice = loss_mask_dice / num_instances
    else:
        z = mask_logits.sum() * 0.0
        loss_mask_bce = z
        loss_mask_dice = z

    # -------------------------
    # (2) Query classification BCE (matched=1, unmatched=0)
    # -------------------------
    class_targets = torch.zeros_like(class_logits, device=device)  # [B, Q]
    for b in range(B):
        pred_ind, _ = indices[b]
        if pred_ind.numel() > 0:
            class_targets[b, pred_ind] = 1.0

    loss_mask_cls = F.binary_cross_entropy_with_logits(class_logits, class_targets)

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

    # -------------------------
    # Presence scalar (differentiable) from per-query class+mask
    #   mask_mass: mean pixel prob per query
    #   qscore: cls_prob * mask_mass
    #   presence_prob: max_q qscore
    # -------------------------
    img_targets = torch.stack([t["image_label"].float() for t in targets]).to(device)  # [B]

    mask_probs = torch.sigmoid(mask_logits)                      # [B, Q, Hm, Wm]
    cls_prob = torch.sigmoid(class_logits)                       # [B, Q]
    mask_mass = mask_probs.flatten(2).mean(-1)                   # [B, Q]
    qscore = cls_prob * mask_mass                                # [B, Q]
    presence_prob = qscore.max(dim=1).values                     # [B]
    presence_prob = presence_prob.clamp(1e-6, 1.0 - 1e-6)

    # -------------------------
    # (3) Presence BCE (image supervision) vs image_label
    # -------------------------
    loss_presence = F.binary_cross_entropy(presence_prob, img_targets)

    if logger is not None:
        logger.debug_event(
            "loss_presence_stats",
            {
                **ctx,
                "presence_mean": float(presence_prob.mean().item()) if B > 0 else 0.0,
                "presence_min": float(presence_prob.min().item()) if B > 0 else 0.0,
                "presence_max": float(presence_prob.max().item()) if B > 0 else 0.0,
                "loss_presence": float(loss_presence.detach().cpu()),
            },
        )

    # -------------------------
    # Authentic suppression (sole suppression term):
    # penalize detection mass on authentic images
    # -------------------------
    authentic_mask = (img_targets == 0.0).to(qscore.dtype)  # [B]
    per_image_penalty = qscore.mean(dim=1)                  # [B]
    penalty = (per_image_penalty * authentic_mask).sum() / max(B, 1)
    loss_auth_penalty = authenticity_penalty_weight * penalty

    if logger is not None:
        # per-query mean "activity" for debugging
        qscore_per_query = qscore.mean(dim=0) if B > 0 else qscore.new_zeros((Q,))
        logger.debug_event(
            "loss_auth_penalty_stats",
            {
                **ctx,
                "authentic_frac": float(authentic_mask.mean().item()) if B > 0 else 0.0,
                "per_image_penalty_mean": float(per_image_penalty.mean().item()) if B > 0 else 0.0,
                "loss_auth_penalty": float(loss_auth_penalty.detach().cpu()),
                "qscore_per_query_mean": float(qscore_per_query.mean().item()) if Q > 0 else 0.0,
                "qscore_per_query_p95": float(qscore_per_query.quantile(0.95).item()) if Q > 0 else 0.0,
                "qscore_per_query_max": float(qscore_per_query.max().item()) if Q > 0 else 0.0,
            },
        )

        # keep the old fg-per-query style log, but now on mask_probs directly
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

    # -------------------------
    # Total (start with all weights = 1)
    # -------------------------
    loss_total = (
        loss_weight_mask_bce * loss_mask_bce
        + loss_weight_mask_dice * loss_mask_dice
        + loss_weight_mask_cls * loss_mask_cls
        + loss_weight_presence * loss_presence
        + loss_weight_auth_penalty * loss_auth_penalty
    )

    return {
        "loss_mask_bce": loss_mask_bce,
        "loss_mask_dice": loss_mask_dice,
        "loss_mask_cls": loss_mask_cls,
        "loss_presence": loss_presence,
        "loss_auth_penalty": loss_auth_penalty,
        "presence_prob": presence_prob.detach(),  # handy for debugging (not used for backprop)
        "loss_total": loss_total,
    }
