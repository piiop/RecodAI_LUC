"""Mask2Former-style instance-segmentation/forgery-detection model.

Combines:
- ConvNeXt+FPN backbone for multi-scale features
- DETR-style transformer decoder with learned queries
- Per-query mask + forgery-classification heads
- Image-level authenticity head
- Hungarian matching, BCE/Dice mask losses, image-auth loss, and an
  authenticity-penalty term for false positives on authentic images

Supports training (returns loss dict) and inference (returns masks,
scores, and image-level forged probability).
""" 

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.ops import FeaturePyramidNetwork

from .losses_metrics import compute_losses

class ConvNeXtFPNBackbone(nn.Module):
    """
    ConvNeXt backbone + FPN neck.
    Default uses the 4 resolution stages: indices (1,3,5,7).
    """
    def __init__(
        self,
        backbone_name="convnext_tiny",
        pretrained=True,
        fpn_out_channels=256,
        out_indices=(1, 3, 5, 7),
        train_backbone=True,
    ):
        super().__init__()
        self.out_indices = out_indices

        # Backbone selector (future-proof for convnext_small/base)
        if backbone_name == "convnext_tiny":
            try:
                backbone = convnext_tiny(
                    weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
                )
            except Exception:
                backbone = convnext_tiny(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.body = backbone.features

        # channel dims for convnext_tiny's blocks
        in_channels_list = [96, 192, 384, 768]

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
        )

        if not train_backbone:
            for p in self.body.parameters():
                p.requires_grad = False

    def forward(self, x):
        feats = []
        out = x
        for i, layer in enumerate(self.body):
            out = layer(out)
            if i in self.out_indices:
                feats.append(out)

        feat_dict = {str(i): f for i, f in enumerate(feats)}
        fpn_out = self.fpn(feat_dict)

        # preserve order 0→3
        return [fpn_out[k] for k in sorted(fpn_out.keys(), key=int)]

class DetrTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def with_pos(self, x, pos):
        if pos is None:
            return x
        return x + pos

    def forward(
        self,
        tgt,                      # [Q, B, C]
        memory,                   # [S, B, C]
        tgt_pos=None,             # [Q, B, C] or None
        memory_pos=None,          # [S, B, C] or None
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # ---- Self-attention (queries attend to themselves) ----
        q = k = self.with_pos(tgt, tgt_pos)
        tgt2, _ = self.self_attn(
            q, k, value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ---- Cross-attention (queries attend to encoder memory) ----
        q = self.with_pos(tgt, tgt_pos)
        k = self.with_pos(memory, memory_pos)
        tgt2, _ = self.multihead_attn(
            q, k, value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ---- Feed-forward ----
        tgt2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
    
class DetrTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # use independent copies of decoder_layer (DETR-style)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,                      # [Q, B, C]
        memory,                   # [S, B, C]
        tgt_pos=None,             # [Q, B, C] or None
        memory_pos=None,          # [S, B, C] or None
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_pos=tgt_pos,
                memory_pos=memory_pos,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate and len(intermediate) > 0:
                intermediate[-1] = output

        if self.return_intermediate:
            # [num_layers, Q, B, C]
            return torch.stack(intermediate)

        # [1, Q, B, C] to match DETR-style interface
        return output.unsqueeze(0)


class SimpleTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_queries=100,
        return_intermediate=True,
    ):
        super().__init__()
        self.num_queries = num_queries

        self.query_embed = nn.Embedding(num_queries, d_model)

        layer = DetrTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.decoder = DetrTransformerDecoder(
            decoder_layer=layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            return_intermediate=return_intermediate,
        )

    def forward(self, feats, pos_list):
        srcs = []
        pos_embs = []

        for feat, pos in zip(feats, pos_list):
            srcs.append(feat.flatten(2).permute(2, 0, 1))   # [S, B, C]
            pos_embs.append(pos.flatten(2).permute(2, 0, 1))

        memory = torch.cat(srcs, dim=0)
        memory_pos = torch.cat(pos_embs, dim=0)

        B = memory.size(1)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(query_pos)

        hs = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_pos=query_pos,
            memory_pos=memory_pos,
        )
        return hs.permute(0, 2, 1, 3)

class PositionEmbeddingSine(nn.Module):
    """
    Standard sine-cosine positional encoding as in DETR.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * torch.pi if scale is None else scale

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        device = x.device

        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)

        y_embed = y_embed.float()
        x_embed = x_embed.float()

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H - 1 + eps) * self.scale
            x_embed = x_embed / (W - 1 + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, 2*num_pos_feats]
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, C, H, W]
        return pos

class Mask2FormerForgeryModel(nn.Module):
    """
    Instance-Seg Transformer (Mask2Former-style) + Authenticity Gate baseline.
    """
    def __init__(
        self,
        num_queries=15,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        mask_dim=256,        
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        backbone_name="convnext_tiny",
        pretrained_backbone=True,
        backbone_trainable=True,
        fpn_out_channels=256,
        authenticity_penalty_weight=5.0,
        auth_gate_forged_threshold=0.5,  
        default_mask_threshold=0.5,      # optional, for masks
        default_cls_threshold=0.5,       # optional, for per-query forgery
        auth_penalty_cls_threshold=None,
        # matching weights
        cost_bce=1.0,
        cost_dice=1.0,
        # loss weights (for convenience total loss)
        loss_weight_mask_bce=1.0,
        loss_weight_mask_dice=1.0,
        loss_weight_mask_cls=1.0,
        loss_weight_img_auth=1.0,
        loss_weight_auth_penalty=1.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.mask_dim = mask_dim
        self.authenticity_penalty_weight = authenticity_penalty_weight

        # matching weights
        self.cost_bce = cost_bce
        self.cost_dice = cost_dice

        # loss weights
        self.loss_weight_mask_bce = loss_weight_mask_bce
        self.loss_weight_mask_dice = loss_weight_mask_dice
        self.loss_weight_mask_cls = loss_weight_mask_cls
        self.loss_weight_img_auth = loss_weight_img_auth
        self.loss_weight_auth_penalty = loss_weight_auth_penalty

        # Backbone + FPN
        self.backbone = ConvNeXtFPNBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            fpn_out_channels=d_model,      # typically equal to transformer dim
            train_backbone=backbone_trainable,
        )
        # Transformer decoder
        self.position_encoding = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.transformer_decoder = SimpleTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_queries=num_queries,
            return_intermediate=True,
        )

        # Pixel decoder: project highest-res FPN level to mask feature space
        self.mask_feature_proj = nn.Conv2d(d_model, mask_dim, kernel_size=1)

        # Instance heads
        self.class_head = nn.Linear(d_model, 1)  # forgery vs ignore, per query
        self.mask_embed_head = nn.Linear(d_model, mask_dim)

        # Image-level authenticity head (global pooled high-level feat)
        self.img_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )
        # Validate / coerce thresholds
        def _coerce_thresh(name, value, default=0.5):
            if value is None:
                print(f"[Warning] `{name}` was None — coercing to default={default}.")
                return default
            return value
        # Thresholds
        self.auth_gate_forged_threshold = auth_gate_forged_threshold
        self.default_mask_threshold = default_mask_threshold
        self.default_cls_threshold = _coerce_thresh(
            "default_cls_threshold", default_cls_threshold
        )
        self.auth_penalty_cls_threshold = _coerce_thresh(
            "auth_penalty_cls_threshold",
            auth_penalty_cls_threshold if auth_penalty_cls_threshold is not None else default_cls_threshold
        )

    def forward(self, images, targets=None, inference_overrides=None):
        """
        images: Tensor [B, 3, H, W]
        targets: list[dict], each with:
          - 'masks': [N_gt, H, W] binary mask tensor
          - 'image_label': scalar tensor 0 (authentic) or 1 (forged)
        Returns:
          - if training: dict of losses
          - if inference: list[dict] with masks, mask_scores, mask_forgery_scores, image_authenticity
        """
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        if not self.training:
            overrides = inference_overrides or {}
            return self.inference(
                mask_logits=mask_logits,
                class_logits=class_logits,
                img_logits=img_logits,
                mask_threshold=overrides.get("mask_threshold", None),
                cls_threshold=overrides.get("cls_threshold", None),
                auth_gate_forged_threshold=overrides.get("auth_gate_forged_threshold", None),
            )    

        B = images.shape[0]

        # Backbone + FPN
        fpn_feats = self.backbone(images)  # [P2, P3, P4, P5]
        # Use highest-res level (P2) for mask features
        mask_feats = self.mask_feature_proj(fpn_feats[0])  # [B, mask_dim, Hm, Wm]
        pos_list = [self.position_encoding(x) for x in fpn_feats]

        # Transformer on multi-scale features
        hs_all = self.transformer_decoder(fpn_feats, pos_list)  # [num_layers, B, Q, C]
        hs = hs_all[-1]                                         # last layer: [B, Q, C]

        # Heads
        class_logits    = self.class_head(hs).squeeze(-1)       # [B, Q]
        mask_embeddings = self.mask_embed_head(hs)              # [B, Q, mask_dim]


        # Produce mask logits via dot-product
        # mask_feats: [B, mask_dim, Hm, Wm]; mask_emb: [B, Q, mask_dim]
        mask_logits = torch.einsum("bqc, bchw -> bqhw", mask_embeddings, mask_feats)

        # Image-level authenticity from highest-level FPN feature (P5)
        high_level_feat = fpn_feats[-1]  # [B, C, Hh, Wh]
        img_logits = self.img_head(high_level_feat)  # [B, 1]
        img_logits = img_logits.squeeze(-1)          # [B]

        if targets is not None:
            return compute_losses(
                mask_logits,
                class_logits,
                img_logits,
                targets,
                cost_bce=self.cost_bce,
                cost_dice=self.cost_dice,
                loss_weight_mask_bce=self.loss_weight_mask_bce,
                loss_weight_mask_dice=self.loss_weight_mask_dice,
                loss_weight_mask_cls=self.loss_weight_mask_cls,
                loss_weight_img_auth=self.loss_weight_img_auth,
                loss_weight_auth_penalty=self.loss_weight_auth_penalty,
                authenticity_penalty_weight=self.authenticity_penalty_weight,
                auth_penalty_cls_threshold=self.auth_penalty_cls_threshold,
            )
        else:
            return self.inference(mask_logits, class_logits, img_logits)

    # ------------------- Losses & matching -------------------
    # (in models/losses_metrics.py)
    # ------------------- Inference -------------------

    def inference(
        self,
        mask_logits,
        class_logits,
        img_logits,
        mask_threshold=None,
        cls_threshold=None,
        auth_gate_forged_threshold=None,
    ):
        """
        Returns list of dicts per image:
          - 'masks': [K, Hm, Wm] uint8
          - 'mask_scores': [K]
          - 'mask_forgery_scores': [K]
          - 'image_authenticity': float in [0,1], prob of "forged"

        Authenticity gate: if prob(forged) < auth_gate_forged_threshold,
        the masks list is empty.
        """
        B, Q, Hm, Wm = mask_logits.shape
        mask_probs = torch.sigmoid(mask_logits)
        cls_probs = torch.sigmoid(class_logits)
        img_probs = torch.sigmoid(img_logits)  # prob "forged"

        # fall back to model defaults
        if mask_threshold is None:
            mask_threshold = self.default_mask_threshold
        if cls_threshold is None:
            cls_threshold = self.default_cls_threshold
        if auth_gate_forged_threshold is None:
            auth_gate_forged_threshold = self.auth_gate_forged_threshold

        outputs = []
        for b in range(B):
            forged_prob = img_probs[b].item()

            # ---- pre-filter stats (always computed) ----
            gate_pass = forged_prob >= auth_gate_forged_threshold
            max_cls_prob = cls_probs[b].max().item() if Q > 0 else 0.0
            num_keep = int((cls_probs[b] > cls_threshold).sum().item()) if Q > 0 else 0
            max_mask_prob = mask_probs[b].max().item() if Q > 0 else 0.0

            # any foreground pixels after mask threshold (pre-keep and post-keep)
            any_fg_pre_keep = bool((mask_probs[b] > mask_threshold).any().item()) if Q > 0 else False

            # gate based directly on forged probability
            if not gate_pass:
                outputs.append({
                    "masks": torch.zeros((0, Hm, Wm), dtype=torch.uint8, device=mask_logits.device),
                    "mask_scores": torch.empty(0, device=mask_logits.device),
                    "mask_forgery_scores": torch.empty(0, device=mask_logits.device),
                    "image_authenticity": forged_prob,

                    # logging / debugging
                    "gate_pass": gate_pass,
                    "max_cls_prob": max_cls_prob,
                    "num_keep": num_keep,
                    "max_mask_prob": max_mask_prob,
                    "any_fg_pre_keep": any_fg_pre_keep,
                    "any_fg_post_keep": False,
                })
                continue

            keep = cls_probs[b] > cls_threshold
            if keep.sum() == 0:
                outputs.append({
                    "masks": torch.zeros((0, Hm, Wm), dtype=torch.uint8, device=mask_logits.device),
                    "mask_scores": torch.empty(0, device=mask_logits.device),
                    "mask_forgery_scores": torch.empty(0, device=mask_logits.device),
                    "image_authenticity": forged_prob,

                    # logging / debugging
                    "gate_pass": gate_pass,
                    "max_cls_prob": max_cls_prob,
                    "num_keep": num_keep,
                    "max_mask_prob": max_mask_prob,
                    "any_fg_pre_keep": any_fg_pre_keep,
                    "any_fg_post_keep": False,
                })
                continue

            masks_b_bool = (mask_probs[b, keep] > mask_threshold)
            any_fg_post_keep = bool(masks_b_bool.any().item())

            masks_b = masks_b_bool.to(torch.uint8)
            scores_b = mask_probs[b, keep].flatten(1).mean(-1)
            cls_b = cls_probs[b, keep]

            outputs.append({
                "masks": masks_b,
                "mask_scores": scores_b,
                "mask_forgery_scores": cls_b,
                "image_authenticity": forged_prob,

                # logging / debugging
                "gate_pass": gate_pass,
                "max_cls_prob": max_cls_prob,
                "num_keep": num_keep,
                "max_mask_prob": max_mask_prob,
                "any_fg_pre_keep": any_fg_pre_keep,
                "any_fg_post_keep": any_fg_post_keep,
            })

        return outputs
