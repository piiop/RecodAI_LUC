"""
Losses and matching utilities for Mask2Former-style forgery model.

Refactor goals (sparse-by-construction, train/inference aligned):
- Train-time top-k query survival (and optional min_mask_mass) applied to *all* losses.
- Few-queries regularizer to make "one strong query" the cheapest solution.
- Sharper presence via smooth-max (log-sum-exp) over per-query activity.
- Balanced class BCE so unmatched queries don't dominate.
- Extra-active queries discouraged via stronger negative weighting (and optional matching bias).
- Rich sparsity metrics logging (per-image + batch aggregates).
"""

from typing import List, Dict, Tuple, Optional

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
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction="none").mean(dim=(1, 2))


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

    return cost_bce * bce + cost_dice * dice


def hungarian_match(
    mask_logits: torch.Tensor,
    targets: List[Dict],
    cost_bce: float = 1.0,
    cost_dice: float = 1.0,
    *,
    allowed_queries: Optional[List[torch.Tensor]] = None,
    extra_query_cost: Optional[List[torch.Tensor]] = None,
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
        allowed_queries: optional list length B of bool masks [Q] (True=eligible)
        extra_query_cost: optional list length B of float costs [Q] added to all rows of cost matrix
                         (useful to bias matching toward/away from certain queries)
        logger: optional logger with .debug_event(tag, payload)
        debug_ctx: optional dict to attach to logs

    Returns:
        indices: list of length B, each (pred_ind, tgt_ind) as LongTensors (pred indices are in original Q space)
    """
    B, Q, Hm, Wm = mask_logits.shape
    device = mask_logits.device
    indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
    ctx = {} if debug_ctx is None else dict(debug_ctx)

    if allowed_queries is None:
        allowed_queries = [torch.ones((Q,), dtype=torch.bool, device=device) for _ in range(B)]
    if extra_query_cost is None:
        extra_query_cost = [None for _ in range(B)]

    for b in range(B):
        tgt_masks = targets[b]["masks"]  # [N_gt, H, W] or empty

        allow = allowed_queries[b]
        if allow is None:
            allow = torch.ones((Q,), dtype=torch.bool, device=device)
        allow = allow.to(device=device, dtype=torch.bool)

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
                    "allowed_q": int(allow.sum().item()),
                },
            )

        if tgt_masks.numel() == 0:
            empty = (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
            indices.append(empty)
            if logger is not None:
                logger.debug_event("hungarian_match_result", {**ctx, "b": b, "matched": 0, "reason": "empty_gt"})
            continue

        # If no queries allowed, no matches.
        if int(allow.sum().item()) == 0:
            empty = (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
            indices.append(empty)
            if logger is not None:
                logger.debug_event("hungarian_match_result", {**ctx, "b": b, "matched": 0, "reason": "no_allowed_queries"})
            continue

        tgt_masks_resized = F.interpolate(
            tgt_masks.unsqueeze(1).float(),
            size=(Hm, Wm),
            mode="nearest",
        ).squeeze(1)  # [N_gt, Hm, Wm]

        pred = mask_logits[b]  # [Q, Hm, Wm]
        pred_allowed_idx = torch.nonzero(allow, as_tuple=False).squeeze(1)  # [Qa]
        pred_allowed = pred[pred_allowed_idx]  # [Qa, Hm, Wm]

        cost = match_cost(pred_allowed, tgt_masks_resized, cost_bce=cost_bce, cost_dice=cost_dice)  # [N_gt, Qa]

        # Optional per-query additive bias (same for all GT rows)
        extra = extra_query_cost[b]
        if extra is not None:
            extra = extra.to(device=device, dtype=cost.dtype)
            extra_allowed = extra[pred_allowed_idx].view(1, -1).expand(cost.shape[0], -1)
            cost = cost + extra_allowed

        tgt_ind_np, pred_ind_np = linear_sum_assignment(cost.detach().cpu().numpy())
        tgt_ind = torch.as_tensor(tgt_ind_np, dtype=torch.long, device=device)
        pred_ind_allowed = torch.as_tensor(pred_ind_np, dtype=torch.long, device=device)

        # Map back to original Q indices
        pred_ind = pred_allowed_idx[pred_ind_allowed]
        indices.append((pred_ind, tgt_ind))

        if logger is not None:
            logger.debug_event(
                "hungarian_match_result",
                {
                    **ctx,
                    "b": b,
                    "cost_shape": [int(cost.shape[0]), int(cost.shape[1])],
                    "matched": int(pred_ind.numel()),
                    "num_gt": int(tgt_masks.shape[0]),
                    "Q": int(Q),
                    "Qa": int(pred_allowed_idx.numel()),
                },
            )

    return indices


# ------------------- Full loss computation -------------------

def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

def compute_losses(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    targets: List[Dict],
    *,
    cost_bce: float = 1.0,
    cost_dice: float = 1.0,
    loss_weight_mask_bce: float = 1.0,
    loss_weight_mask_dice: float = 1.0,
    loss_weight_mask_cls: float = 1.0,
    loss_weight_presence: float = 1.0,
    loss_weight_auth_penalty: float = 1.0,
    authenticity_penalty_weight: float = 1.0,
    # --- sparse-by-construction knobs (train-time) ---
    train_topk: int = 2,
    train_min_mask_mass: float = 0.0,
    # extra matching bias (optional): encourage matching to high-activity queries
    # (implemented as additive cost = cost_qscore * (1 - qscore))
    cost_qscore: float = 0.0,
    # --- class BCE balancing / discouraging extras ---
    cls_neg_pos_ratio: int = 8,          # max negatives per positive (per-image); if pos==0, fall back to topk-only
    cls_neg_weight: float = 0.25,        # base weight for negatives (unmatched)
    cls_unmatched_multiplier: float = 2.0,  # extra discouragement for unmatched actives (still negatives)
    # --- regularizers ---
    few_queries_lambda: float = 0.10,    # λ * qscore.sum(dim=1).mean()
    # --- presence smoothing ---
    presence_lse_beta: float = 10.0,     # smooth-max in logit(qscore) space
    tv_lambda: float = 0.0,
    # --- logging ---
    sparsity_thresholds: Tuple[float, ...] = (0.05, 0.10, 0.20),
    logger=None,
    debug_ctx: Dict | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Refactored losses (no image head supervision; no thresholds/temperature in losses).

    Core:
      1) Instance mask loss on matched pairs (Dice + BCE)
      2) Query classification BCE (matched=1, unmatched=0), balanced
      3) Presence BCE vs image_label using a differentiable smooth-max presence_prob

    Sparse-by-construction:
      - Compute qscore = cls_prob * mask_mass
      - Select allowed queries per image using (top-k by qscore) AND (mask_mass >= min_mask_mass)
      - Apply this same survival rule to *matching*, *mask loss*, and *class loss* weighting

    Suppression:
      - loss_auth_penalty: penalize activity on authentic images (mean qscore)

    Regularizer:
      - loss_few_queries: penalize total activity to encourage a single winner

    Returns keys are kept stable for training scripts:
      loss_mask_bce, loss_mask_dice, loss_mask_cls, loss_presence, loss_auth_penalty, loss_total
    """
    B, Q, Hm, Wm = mask_logits.shape
    device = mask_logits.device
    ctx = {} if debug_ctx is None else dict(debug_ctx)

    # -------------------------
    # Train-aligned activity signals
    # -------------------------
    mask_probs = torch.sigmoid(mask_logits)                      # [B,Q,Hm,Wm]
    cls_prob = torch.sigmoid(class_logits)                       # [B,Q]
    mask_mass = mask_probs.flatten(2).mean(-1)                   # [B,Q] in [0,1]
    qscore = cls_prob * mask_mass                                # [B,Q] in [0,1]

    # -------------------------
    # Train-time survival: top-k + min_mask_mass (align with inference)
    # -------------------------
    k = int(train_topk) if train_topk is not None else 0
    k = max(k, 0)
    allowed_queries: List[torch.Tensor] = []
    extra_query_cost: List[Optional[torch.Tensor]] = []

    for b in range(B):
        allow = torch.zeros((Q,), dtype=torch.bool, device=device)

        if k > 0:
            kk = min(k, Q)
            top_idx = torch.topk(qscore[b], k=kk, largest=True).indices
            allow[top_idx] = True
        else:
            # if k==0, allow all (but still can be filtered by min_mask_mass below)
            allow[:] = True

        if train_min_mask_mass is not None and float(train_min_mask_mass) > 0.0:
            allow = allow & (mask_mass[b] >= float(train_min_mask_mass))

        allowed_queries.append(allow)

        if cost_qscore is not None and float(cost_qscore) > 0.0:
            # lower cost for higher qscore; additive cost must be >=0
            extra_query_cost.append(float(cost_qscore) * (1.0 - qscore[b]).detach())
        else:
            extra_query_cost.append(None)

    # -------------------------
    # Hungarian matching (restricted to allowed queries)
    # -------------------------
    indices = hungarian_match(
        mask_logits,
        targets,
        cost_bce=cost_bce,
        cost_dice=cost_dice,
        allowed_queries=allowed_queries,
        extra_query_cost=extra_query_cost,
        logger=logger,
        debug_ctx=debug_ctx,
    )

    # -------------------------
    # (1) Mask losses (Dice + BCE) on matched pairs ONLY (and only from allowed queries via matching)
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

        pred_masks = mask_logits[b, pred_ind]        # [M,Hm,Wm]
        gt_masks = tgt_masks_resized[tgt_ind]        # [M,Hm,Wm]

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
    # (2) Query classification BCE (balanced + top-k aligned)
    #   - matched queries (from restricted matching) are positives
    #   - all others are negatives, BUT:
    #       * we downweight negatives
    #       * optionally subsample negatives per image so they don't dominate
    #       * apply extra penalty to "unmatched but allowed/active" negatives
    # -------------------------
    class_targets = torch.zeros_like(class_logits, device=device)  # [B,Q]
    for b in range(B):
        pred_ind, _ = indices[b]
        if pred_ind.numel() > 0:
            class_targets[b, pred_ind] = 1.0

    # Build per-element weights
    weights = torch.ones_like(class_logits, device=device, dtype=mask_logits.dtype)

    # default: negatives are downweighted
    weights = torch.where(class_targets > 0.5, weights, weights * float(cls_neg_weight))

    # extra: if a query survived (allowed) but is still unmatched => penalize more (discourage extras)
    if cls_unmatched_multiplier is not None and float(cls_unmatched_multiplier) != 1.0:
        for b in range(B):
            allow = allowed_queries[b]
            unmatched_active = allow & (class_targets[b] < 0.5)
            if unmatched_active.any():
                weights[b, unmatched_active] = weights[b, unmatched_active] * float(cls_unmatched_multiplier)

    # optional: negative subsampling per image (keep all positives, cap negatives)
    if cls_neg_pos_ratio is not None and int(cls_neg_pos_ratio) > 0:
        neg_pos_ratio = int(cls_neg_pos_ratio)
        for b in range(B):
            pos_idx = torch.nonzero(class_targets[b] > 0.5, as_tuple=False).squeeze(1)
            neg_idx = torch.nonzero(class_targets[b] < 0.5, as_tuple=False).squeeze(1)

            npos = int(pos_idx.numel())
            nneg = int(neg_idx.numel())
            if nneg == 0:
                continue

            # if no positives, only keep negatives among allowed (top-k / min-mass) so cls loss stays focused
            if npos == 0:
                allow = allowed_queries[b]
                keep_neg = torch.nonzero(allow & (class_targets[b] < 0.5), as_tuple=False).squeeze(1)
                drop_neg = torch.nonzero((~allow) & (class_targets[b] < 0.5), as_tuple=False).squeeze(1)
                if drop_neg.numel() > 0:
                    weights[b, drop_neg] = 0.0
                # If even allowed has 0 (k=0 or min-mass filtered), do nothing.
                _ = keep_neg
                continue

            max_neg = max(npos * neg_pos_ratio, npos)  # at least npos
            if nneg > max_neg:
                # Keep a random subset of negatives; zero weight for the rest.
                perm = torch.randperm(nneg, device=device)
                keep = neg_idx[perm[:max_neg]]
                drop = neg_idx[perm[max_neg:]]
                weights[b, drop] = 0.0
                _ = keep

    # Weighted BCE on logits
    per_elem = F.binary_cross_entropy_with_logits(class_logits, class_targets, reduction="none")
    denom = weights.sum().clamp_min(1.0)
    loss_mask_cls = (per_elem * weights).sum() / denom

    if logger is not None:
        pos = int(class_targets.sum().item())
        tot = int(class_targets.numel())
        wsum = float(weights.sum().detach().cpu())
        logger.debug_event(
            "loss_cls_targets",
            {
                **ctx,
                "B": int(B),
                "Q": int(Q),
                "pos": pos,
                "total": tot,
                "pos_frac": (pos / max(tot, 1)),
                "weights_sum": wsum,
                "weights_mean": float((weights.mean().detach().cpu())) if weights.numel() else 0.0,
            },
        )

    # -------------------------
    # Presence scalar (sharpened): smooth max via LSE in logit(qscore) space
    #   presence_logit = (1/beta) * logsumexp(beta * logit(qscore))
    #   presence_prob  = sigmoid(presence_logit)
    # -------------------------
    img_targets = torch.stack([t["image_label"].float() for t in targets]).to(device)  # [B]

    beta = float(presence_lse_beta) if presence_lse_beta is not None else 0.0
    if beta <= 0.0:
        # fallback to hard max (still stable)
        presence_prob = qscore.max(dim=1).values
    else:
        qlogit = _safe_logit(qscore)  # [B,Q]
        presence_logit = torch.logsumexp(beta * qlogit, dim=1) / beta  # [B]
        presence_prob = torch.sigmoid(presence_logit)

    presence_prob = presence_prob.clamp(1e-6, 1.0 - 1e-6)

        # -------------------------
    # NEW: TV / boundary smoothness penalty
    #   - computed on mask_probs (sigmoid(mask_logits))
    #   - only on allowed/surviving queries
    #   - only on forged images (image_label==1)
    # -------------------------
    loss_tv = mask_logits.new_zeros(())
    if tv_lambda is not None and float(tv_lambda) > 0.0:
        tv_sum = mask_logits.new_zeros(())
        tv_count = 0

        for b in range(B):
            if float(img_targets[b].item()) < 0.5:
                continue  # forged-only
            keep = allowed_queries[b]
            if keep is None or int(keep.sum().item()) == 0:
                continue

            p = mask_probs[b, keep]  # [K,H,W]
            if p.numel() == 0:
                continue

            # anisotropic TV (L1) averaged over pixels and queries
            tv_h = (p[:, :, 1:] - p[:, :, :-1]).abs().mean()
            tv_v = (p[:, 1:, :] - p[:, :-1, :]).abs().mean()
            tv = tv_h + tv_v

            tv_sum = tv_sum + tv
            tv_count += 1

        if tv_count > 0:
            loss_tv = tv_sum / tv_count
        else:
            loss_tv = mask_logits.sum() * 0.0

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
                "presence_lse_beta": beta,
            },
        )

    # -------------------------
    # Few-queries regularizer: penalize total activity (encourage one winner)
    # -------------------------
    loss_few_queries = qscore.sum(dim=1).mean()  # [B] -> scalar

    # -------------------------
    # Authentic suppression (sole suppression term):
    # penalize activity on authentic images
    # -------------------------
    authentic_mask = (img_targets == 0.0).to(qscore.dtype)  # [B]
    per_image_penalty = qscore.mean(dim=1)                  # [B]
    penalty = (per_image_penalty * authentic_mask).sum() / max(B, 1)
    loss_auth_penalty = authenticity_penalty_weight * penalty

    # -------------------------
    # Sparsity monitoring logs
    # -------------------------
    if logger is not None:
        with torch.no_grad():
            # per-image counts above thresholds
            per_img = {}
            for thr in sparsity_thresholds:
                thr_f = float(thr)
                per_img[f"num_q_above_{thr_f:g}"] = (qscore > thr_f).sum(dim=1).detach().cpu().tolist()

            qsum = qscore.sum(dim=1).clamp_min(1e-12)
            qmax = qscore.max(dim=1).values
            qmax_over_sum = (qmax / qsum).detach()

            mm_max = mask_mass.max(dim=1).values.detach()

            allow_counts = torch.stack([a.to(torch.int64) for a in allowed_queries], dim=0).sum(dim=1) if B > 0 else torch.zeros((0,), dtype=torch.int64)

            logger.debug_event(
                "sparsity_metrics",
                {
                    **ctx,
                    "train_topk": int(train_topk),
                    "train_min_mask_mass": float(train_min_mask_mass) if train_min_mask_mass is not None else None,
                    "allowed_queries_per_image": allow_counts.detach().cpu().tolist(),
                    "qscore_max": qmax.detach().cpu().tolist(),
                    "qscore_sum": qsum.detach().cpu().tolist(),
                    "qscore_max_over_sum": qmax_over_sum.detach().cpu().tolist(),
                    "mask_mass_max": mm_max.detach().cpu().tolist(),
                    "batch": {
                        "allowed_mean": float(allow_counts.float().mean().item()) if allow_counts.numel() else 0.0,
                        "qmax_over_sum_mean": float(qmax_over_sum.mean().item()) if qmax_over_sum.numel() else 0.0,
                        "mask_mass_max_mean": float(mm_max.mean().item()) if mm_max.numel() else 0.0,
                    },
                    "per_image": per_img,
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

        logger.debug_event(
            "loss_auth_penalty_stats",
            {
                **ctx,
                "authentic_frac": float(authentic_mask.mean().item()) if B > 0 else 0.0,
                "per_image_penalty_mean": float(per_image_penalty.mean().item()) if B > 0 else 0.0,
                "loss_auth_penalty": float(loss_auth_penalty.detach().cpu()),
                "few_queries_lambda": float(few_queries_lambda),
                "loss_few_queries": float(loss_few_queries.detach().cpu()),
            },
        )

    # -------------------------
    # Total
    # -------------------------
    loss_total = (
        loss_weight_mask_bce * loss_mask_bce
        + loss_weight_mask_dice * loss_mask_dice
        + loss_weight_mask_cls * loss_mask_cls
        + loss_weight_presence * loss_presence
        + loss_weight_auth_penalty * loss_auth_penalty
        + float(few_queries_lambda) * loss_few_queries
        + float(tv_lambda) * loss_tv
    )

    return {
        "loss_mask_bce": loss_mask_bce,
        "loss_mask_dice": loss_mask_dice,
        "loss_mask_cls": loss_mask_cls,
        "loss_presence": loss_presence,
        "loss_auth_penalty": loss_auth_penalty,
        "loss_tv": loss_tv,
        "presence_prob": presence_prob.detach(),
        "loss_total": loss_total,
    }
