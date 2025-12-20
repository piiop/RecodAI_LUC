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
import math
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.ops import FeaturePyramidNetwork

from .losses_metrics import compute_losses

def _get_level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
    # spatial_shapes: [L, 2] (H, W)
    level_start_index = torch.zeros((spatial_shapes.size(0),), dtype=torch.long, device=spatial_shapes.device)
    level_start_index[1:] = torch.cumsum(spatial_shapes[:, 0] * spatial_shapes[:, 1], dim=0)[:-1]
    return level_start_index


def _build_reference_points(spatial_shapes: torch.Tensor, device) -> torch.Tensor:
    """
    Build reference points for each position in each level.
    Returns: [1, sum(HW), 2] in normalized [0,1] coords (x,y).
    """
    ref_list = []
    for (H, W) in spatial_shapes.tolist():
        y, x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device),
            torch.linspace(0.5, W - 0.5, W, device=device),
            indexing="ij",
        )
        ref = torch.stack((x / W, y / H), dim=-1)  # [H, W, 2]
        ref_list.append(ref.reshape(-1, 2))
    return torch.cat(ref_list, dim=0).unsqueeze(0)  # [1, S, 2]


class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention (encoder flavor).
    Pure PyTorch reference implementation (grid_sample), no CUDA ops.
    """
    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.d_per_head = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # small radial init for offsets (like common refs)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # [H,2]
        grid = grid / grid.abs().max(dim=-1, keepdim=True).values  # normalize
        grid = grid.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for p in range(self.n_points):
            grid[:, :, p, :] *= (p + 1)
        self.sampling_offsets.bias = nn.Parameter(grid.reshape(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(self, query, value, spatial_shapes, level_start_index, reference_points):
        """
        query: [B, S, C]
        value: [B, S, C]
        spatial_shapes: [L, 2] (H,W)
        level_start_index: [L]
        reference_points: [B, S, 2] in [0,1] (x,y)
        """
        B, S, C = query.shape
        L = spatial_shapes.size(0)
        assert L == self.n_levels

        value = self.value_proj(value)
        value = value.view(B, S, self.n_heads, self.d_per_head)

        sampling_offsets = self.sampling_offsets(query).view(
            B, S, self.n_heads, self.n_levels, self.n_points, 2
        )
        attn_weights = self.attention_weights(query).view(
            B, S, self.n_heads, self.n_levels, self.n_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Normalize offsets by (W,H) per level
        spatial_shapes_f = spatial_shapes.to(query.dtype)  # [L,2]
        # [1,1,1,L,1,2] -> (W,H) ordering for x,y
        normalizer = torch.stack([spatial_shapes_f[:, 1], spatial_shapes_f[:, 0]], dim=-1)
        normalizer = normalizer.view(1, 1, 1, L, 1, 2)

        # reference_points: [B,S,2] -> [B,S,1,L,1,2]
        ref = reference_points[:, :, None, :, None, :]
        sampling_locations = ref + sampling_offsets / normalizer  # in [0,1] approx

        # Sample per level via grid_sample
        output = torch.zeros((B, S, self.n_heads, self.d_per_head), device=query.device, dtype=query.dtype)

        for lvl in range(L):
            H, W = spatial_shapes[lvl].tolist()
            start = level_start_index[lvl].item()
            end = start + H * W

            # (B, HW, nH, d) -> (B, nH, d, H, W)
            value_l = value[:, start:end].permute(0, 2, 3, 1).contiguous()
            value_l = value_l.view(B, self.n_heads, self.d_per_head, H, W)

            # grid_sample expects grid in [-1,1], with last dim (x,y)
            grid = sampling_locations[:, :, :, lvl, :, :]  # [B,S,nH,nP,2] in [0,1]
            grid = grid * 2.0 - 1.0  # -> [-1,1]
            # reshape to sample all points: (B*nH, S*nP, 1, 2)
            grid = grid.permute(0, 2, 1, 3, 4).contiguous()  # [B,nH,S,nP,2]
            grid = grid.view(B * self.n_heads, S * self.n_points, 1, 2)

            # sample: input (B*nH, d, H, W), grid (B*nH, outH, outW, 2)
            sampled = F.grid_sample(
                value_l.view(B * self.n_heads, self.d_per_head, H, W),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B*nH, d, S*nP, 1]
            sampled = sampled.view(B, self.n_heads, self.d_per_head, S, self.n_points)
            sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()  # [B,S,nH,nP,d]

            w = attn_weights[:, :, :, lvl, :].unsqueeze(-1)  # [B,S,nH,nP,1]
            output = output + (sampled * w).sum(dim=3)  # sum over nP -> [B,S,nH,d]

        output = output.view(B, S, C)
        output = self.output_proj(output)
        return output


class MSDeformAttnEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_heads, n_levels, n_points)

        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, src, pos, spatial_shapes, level_start_index, reference_points):
        # src,pos: [B,S,C]
        q = src + pos
        src2 = self.self_attn(q, src, spatial_shapes, level_start_index, reference_points)
        src = self.norm1(src + self.drop1(src2))

        src2 = self.linear2(self.drop_ffn(F.relu(self.linear1(src))))
        src = self.norm2(src + self.drop2(src2))
        return src


class MSDeformAttnEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else type(layer)(**layer.__dict__["_modules"] == {} and {}) for i in range(num_layers)])
        # NOTE: above line is brittle; we’ll use deepcopy instead:
        self.layers = nn.ModuleList([torch.nn.modules.module.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src, pos, spatial_shapes, level_start_index, reference_points):
        out = src
        for l in self.layers:
            out = l(out, pos, spatial_shapes, level_start_index, reference_points)
        return out


class Mask2FormerPixelDecoder(nn.Module):
    """
    Pixel decoder:
    - takes 4 FPN levels [P2,P3,P4,P5] (high->low res)
    - builds 3 levels for deformable encoder (P3,P4,P5 by default)
    - runs MSDeformAttn encoder
    - top-down fuses into high-res mask_features at P2 resolution
    Returns:
      memory_feats: list of 3 tensors for transformer decoder (high->low res)
      mask_features: [B, mask_dim, H2, W2]
    """
    def __init__(
        self,
        in_channels=256,
        conv_dim=256,
        mask_dim=256,
        n_levels=3,              # encoder levels
        n_heads=8,
        n_points=4,
        enc_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.n_levels = n_levels

        # project all 4 FPN levels to conv_dim
        self.input_proj = nn.ModuleList([nn.Conv2d(in_channels, conv_dim, 1) for _ in range(4)])

        # level embeddings for encoder levels (P3,P4,P5)
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, conv_dim))
        nn.init.normal_(self.level_embed, std=0.02)

        enc_layer = MSDeformAttnEncoderLayer(
            d_model=conv_dim,
            n_heads=n_heads,
            n_levels=n_levels,
            n_points=n_points,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.ModuleList([copy.deepcopy(enc_layer) for _ in range(enc_layers)])

        # top-down fusion convs (P5->P4->P3->P2)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(conv_dim, conv_dim, 1),  # P3
            nn.Conv2d(conv_dim, conv_dim, 1),  # P4
            nn.Conv2d(conv_dim, conv_dim, 1),  # P5
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, padding=1), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, padding=1), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, padding=1), nn.ReLU(inplace=True)),
        ])

        self.mask_out = nn.Conv2d(conv_dim, mask_dim, 1)

    def forward(self, fpn_feats, pos_encoding_fn):
        """
        fpn_feats: [P2,P3,P4,P5] (high->low res)
        pos_encoding_fn: callable that returns [B,C,H,W] pos enc for a feature map
        """
        p2, p3, p4, p5 = [proj(f) for proj, f in zip(self.input_proj, fpn_feats)]

        # encoder levels: use P3,P4,P5 (3 levels)
        enc_feats = [p3, p4, p5]
        B = p3.size(0)

        spatial_shapes = torch.tensor([(f.size(2), f.size(3)) for f in enc_feats], device=p3.device, dtype=torch.long)  # [L,2]
        level_start_index = _get_level_start_index(spatial_shapes)  # [L]
        ref = _build_reference_points(spatial_shapes, device=p3.device)  # [1, S, 2]
        L = spatial_shapes.size(0)
        reference_points = ref.unsqueeze(2).repeat(B, 1, L, 1)  # [B, S, L, 2]

        # flatten + add level embeddings
        src_list = []
        pos_list = []
        for lvl, feat in enumerate(enc_feats):
            pos = pos_encoding_fn(feat)  # [B,C,H,W]
            H, W = feat.shape[-2:]
            src = feat.flatten(2).transpose(1, 2)  # [B,HW,C]
            pos = pos.flatten(2).transpose(1, 2)  # [B,HW,C]
            pos = pos + self.level_embed[lvl].view(1, 1, -1)
            src_list.append(src)
            pos_list.append(pos)

        src = torch.cat(src_list, dim=1)  # [B,S,C]
        pos = torch.cat(pos_list, dim=1)  # [B,S,C]

        # deformable encoder
        out = src
        for layer in self.encoder:
            out = layer(out, pos, spatial_shapes, level_start_index, reference_points)

        # split back into levels
        outs = []
        cursor = 0
        for (H, W) in spatial_shapes.tolist():
            n = H * W
            o = out[:, cursor:cursor+n, :].transpose(1, 2).contiguous().view(B, self.conv_dim, H, W)
            outs.append(o)
            cursor += n

        # memory feats for transformer decoder: return high->low res (P3,P4,P5)
        mem_p3, mem_p4, mem_p5 = outs[0], outs[1], outs[2]
        memory_feats = [mem_p3, mem_p4, mem_p5]

        # top-down fusion into P2-sized mask feature
        # start from mem_p5 -> mem_p4 -> mem_p3, then fuse into p2
        x5 = self.lateral_convs[2](mem_p5)
        x4 = self.lateral_convs[1](mem_p4) + F.interpolate(x5, size=mem_p4.shape[-2:], mode="bilinear", align_corners=False)
        x4 = self.output_convs[1](x4)

        x3 = self.lateral_convs[0](mem_p3) + F.interpolate(x4, size=mem_p3.shape[-2:], mode="bilinear", align_corners=False)
        x3 = self.output_convs[0](x3)

        # fuse into P2 (projected p2) as final high-res feature
        x2 = p2 + F.interpolate(x3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        mask_features = self.mask_out(x2)  # [B, mask_dim, H2, W2]

        return memory_feats, mask_features

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
    # inference thresholds
    default_mask_threshold=0.0,
    default_cls_threshold=0.0,  # legacy (kept for backwards compat in inference)
    # train-aligned inference knobs
    default_qscore_threshold=None,
    default_topk=None,
    default_min_mask_mass=None,
    default_presence_threshold=None,
    # matching weights
    cost_bce=1.0,
    cost_dice=1.0,
    cost_qscore=0.0,
    # loss weights (for convenience total loss)
    authenticity_penalty_weight=5.0,
    loss_weight_mask_bce=1.0,
    loss_weight_mask_dice=1.0,
    loss_weight_mask_cls=1.0,
    loss_weight_presence=1.0,
    loss_weight_auth_penalty=1.0,
    # sparsity / regularization
    few_queries_lambda=0.10,
    presence_lse_beta=10.0,
    # class BCE balancing / discourage extras
    cls_neg_pos_ratio=8,
    cls_neg_weight=0.25,
    cls_unmatched_multiplier=2.0,
    # logging
    sparsity_thresholds=(0.05, 0.10, 0.20),
):
    super().__init__()

    # -----------------------------
    # Core config
    # -----------------------------
    self.num_queries = num_queries
    self.d_model = d_model
    self.mask_dim = mask_dim
    self.authenticity_penalty_weight = authenticity_penalty_weight

    # -----------------------------
    # Thresholds / inference knobs
    # -----------------------------
    self.default_mask_threshold = default_mask_threshold
    self.default_cls_threshold = self._coerce_thresh(
        "default_cls_threshold", default_cls_threshold
    )
    self.default_qscore_threshold = default_qscore_threshold
    self.default_topk = default_topk
    self.default_min_mask_mass = default_min_mask_mass
    self.default_presence_threshold = default_presence_threshold

    # -----------------------------
    # Matching weights
    # -----------------------------
    self.cost_bce = cost_bce
    self.cost_dice = cost_dice
    self.cost_qscore = cost_qscore

    # -----------------------------
    # Loss weights
    # -----------------------------
    self.loss_weight_mask_bce = loss_weight_mask_bce
    self.loss_weight_mask_dice = loss_weight_mask_dice
    self.loss_weight_mask_cls = loss_weight_mask_cls
    self.loss_weight_presence = loss_weight_presence
    self.loss_weight_auth_penalty = loss_weight_auth_penalty

    # -----------------------------
    # Sparsity / regularization knobs
    # -----------------------------
    self.few_queries_lambda = few_queries_lambda
    self.presence_lse_beta = presence_lse_beta
    self.cls_neg_pos_ratio = cls_neg_pos_ratio
    self.cls_neg_weight = cls_neg_weight
    self.cls_unmatched_multiplier = cls_unmatched_multiplier
    self.sparsity_thresholds = sparsity_thresholds

    # -----------------------------
    # Backbone + FPN
    # -----------------------------
    self.backbone = ConvNeXtFPNBackbone(
        backbone_name=backbone_name,
        pretrained=pretrained_backbone,
        fpn_out_channels=fpn_out_channels,
        train_backbone=backbone_trainable,
    )

    # -----------------------------
    # Transformer decoder + positional enc
    # -----------------------------
    self.position_encoding = PositionEmbeddingSine(
        fpn_out_channels // 2, normalize=True
    )
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

    # -----------------------------
    # Pixel decoder / mask feature projection
    # -----------------------------
    self.pixel_decoder = Mask2FormerPixelDecoder(
        in_channels=fpn_out_channels,   # 256 from your FPN
        conv_dim=d_model,               # 256
        mask_dim=mask_dim,              # 256
        n_levels=3,                     # use P3,P4,P5 for deform encoder
        n_heads=nhead,
        n_points=4,
        enc_layers=6,                   # typical
        dim_feedforward=1024,
        dropout=dropout,
    )

    # -----------------------------
    # Instance heads
    # -----------------------------
    self.class_head = nn.Linear(d_model, 1)  # forgery vs ignore, per query
    self.mask_embed_head = nn.Linear(d_model, mask_dim)


    def forward(self, images, targets=None, inference_overrides=None):
        """
        images: Tensor [B, 3, H, W] or list[Tensor(C,H,W)]
        targets: list[dict] or None

        Train/inference alignment:
        qscore[q]      = sigmoid(class_logit[q]) * mean(sigmoid(mask_logit[q]))
        presence_prob  = max_q qscore[q]
        """
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        # Backbone + pixel decoder
        fpn_feats = self.backbone(images)  # [P2, P3, P4, P5]
        memory_feats, mask_feats = self.pixel_decoder(fpn_feats, self.position_encoding)

        # Build positional encodings per feature level (match forward_logits / decoder API)
        pos_list = [self.position_encoding(x) for x in memory_feats]

        # Transformer decoder (take last layer output)
        hs = self.transformer_decoder(memory_feats, pos_list)[-1]  # [B, Q, C]

        # Heads
        class_logits = self.class_head(hs).squeeze(-1)  # [B, Q]
        mask_embeddings = self.mask_embed_head(hs)      # [B, Q, mask_dim]
        mask_logits = torch.einsum("bqc, bchw -> bqhw", mask_embeddings, mask_feats)  # [B,Q,Hm,Wm]

        overrides = inference_overrides or {}

        # ---- training loss ----
        if self.training and targets is not None:
            # Train-time survival should mirror inference rules (top-k + min-mass),
            # but allow per-call overrides.
            train_topk = overrides.get("topk", None)
            if train_topk is None:
                train_topk = self.default_topk
            if train_topk is None:
                train_topk = 2

            train_min_mask_mass = overrides.get("min_mask_mass", None)
            if train_min_mask_mass is None:
                train_min_mask_mass = self.default_min_mask_mass
            if train_min_mask_mass is None:
                train_min_mask_mass = 0.0

            return compute_losses(
                mask_logits,
                class_logits,
                targets,
                # matching
                cost_bce=self.cost_bce,
                cost_dice=self.cost_dice,
                cost_qscore=float(getattr(self, "cost_qscore", 0.0)),
                # loss weights (map legacy img_auth -> presence)
                authenticity_penalty_weight=self.authenticity_penalty_weight,
                loss_weight_mask_bce=self.loss_weight_mask_bce,
                loss_weight_mask_dice=self.loss_weight_mask_dice,
                loss_weight_mask_cls=self.loss_weight_mask_cls,
                loss_weight_presence=getattr(self, "loss_weight_presence", self.loss_weight_img_auth),
                loss_weight_auth_penalty=self.loss_weight_auth_penalty,
                # sparse-by-construction knobs
                train_topk=int(train_topk),
                train_min_mask_mass=float(train_min_mask_mass),
                few_queries_lambda=float(getattr(self, "few_queries_lambda", 0.10)),
                presence_lse_beta=float(getattr(self, "presence_lse_beta", 10.0)),
                # class BCE balancing / discourage extras
                cls_neg_pos_ratio=int(getattr(self, "cls_neg_pos_ratio", 8)),
                cls_neg_weight=float(getattr(self, "cls_neg_weight", 0.25)),
                cls_unmatched_multiplier=float(getattr(self, "cls_unmatched_multiplier", 2.0)),
                # logging
                sparsity_thresholds=tuple(getattr(self, "sparsity_thresholds", (0.05, 0.10, 0.20))),
                logger=overrides.get("logger", None),
                debug_ctx=overrides.get("debug_ctx", None),
            )

        # ---- inference (also used when targets is None) ----
        preds = self.inference(
            mask_logits=mask_logits,
            class_logits=class_logits,
            mask_threshold=overrides.get("mask_threshold", None),
            cls_threshold=overrides.get("cls_threshold", None),               # legacy
            qscore_threshold=overrides.get("qscore_threshold", None),         # preferred
            topk=overrides.get("topk", None),
            min_mask_mass=overrides.get("min_mask_mass", None),
            presence_threshold=overrides.get("presence_threshold", None),
        )

        # return raw logits/probs for debugging/analysis
        if overrides.get("return_logits", False) or overrides.get("return_raw", False):
            # Compute train-aligned presence/activity for callers that want tensors back
            mask_probs = torch.sigmoid(mask_logits)
            cls_probs = torch.sigmoid(class_logits if class_logits.dim() == 2 else class_logits.squeeze(-1))
            mask_mass = mask_probs.flatten(2).mean(-1)   # [B,Q]
            qscore = cls_probs * mask_mass               # [B,Q]
            presence_prob = qscore.max(dim=1).values     # [B]

            return {
                "preds": preds,
                "mask_logits": mask_logits,
                "class_logits": class_logits,
                "mask_probs": mask_probs,
                "cls_probs": cls_probs,
                "mask_mass": mask_mass,
                "qscore": qscore,
                "presence_prob": presence_prob,
            }

        return preds


    @torch.no_grad()
    def forward_logits(self, images):
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        fpn_feats = self.backbone(images)
        memory_feats, mask_feats = self.pixel_decoder(fpn_feats, self.position_encoding)
        pos_list = [self.position_encoding(x) for x in memory_feats]
        hs = self.transformer_decoder(memory_feats, pos_list)[-1]

        class_logits = self.class_head(hs).squeeze(-1)  # [B,Q]
        mask_embeddings = self.mask_embed_head(hs)       # [B,Q,C]
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embeddings, mask_feats)
        return mask_logits, class_logits

    def inference(
        self,
        mask_logits,
        class_logits,
        mask_threshold=None,
        cls_threshold=None,          # legacy (kept for backwards compat)
        qscore_threshold=None,       # threshold on (cls_prob * mask_mass)
        topk=None,                  # keep top-k queries by qscore (per image)
        min_mask_mass=None,         # additional filter on mask_mass (fraction of pixels)
        presence_threshold=None,     # optional “gate” threshold on presence_prob
    ):
        """
        Returns list of dicts per image:
        - 'masks': [K, Hm, Wm] uint8
        - 'mask_scores': [K]          (qscore)
        - 'mask_forgery_scores': [K]  (cls_prob)
        - 'image_authenticity': float in [0,1], prob of "forged" (presence_prob)

        Train/Inference consistency:
        qscore[q]   = sigmoid(class_logit[q]) * mean(sigmoid(mask_logit[q]))
        presence    = max_q qscore[q]
        """
        import torch

        B, Q, Hm, Wm = mask_logits.shape

        # Defaults (kept compatible with existing config knobs if present)
        if mask_threshold is None:
            mask_threshold = getattr(self, "default_mask_threshold", 0.5)

        # Prefer qscore thresholding by default (train-aligned)
        if qscore_threshold is None:
            qscore_threshold = getattr(self, "default_qscore_threshold", None)

        if topk is None:
            topk = getattr(self, "default_topk", None)

        if min_mask_mass is None:
            min_mask_mass = getattr(self, "default_min_mask_mass", None)

        if presence_threshold is None:
            presence_threshold = getattr(self, "default_presence_threshold", None)

        # Shapes: allow class_logits to be [B,Q,1]
        if class_logits.dim() == 3 and class_logits.size(-1) == 1:
            class_logits = class_logits.squeeze(-1)

        mask_probs = torch.sigmoid(mask_logits)         # [B,Q,H,W]
        cls_probs = torch.sigmoid(class_logits)         # [B,Q]

        # mask_mass in [0,1]: mean probability mass across pixels
        mask_mass = mask_probs.flatten(2).mean(-1)      # [B,Q]
        qscore = cls_probs * mask_mass                  # [B,Q]
        presence_prob = qscore.max(dim=1).values        # [B]

        outputs = []
        for b in range(B):
            # --- robust logging / debugging (kept + expanded) ---
            max_cls_prob = float(cls_probs[b].max().detach().cpu())
            max_mask_prob = float(mask_probs[b].max().detach().cpu())
            max_qscore = float(qscore[b].max().detach().cpu())
            any_fg_pre_keep = bool((mask_probs[b] > mask_threshold).any().detach().cpu())
            mean_mask_mass = float(mask_mass[b].mean().detach().cpu())
            max_mask_mass = float(mask_mass[b].max().detach().cpu())
            image_presence = float(presence_prob[b].detach().cpu())

            # Selection: topk by qscore OR threshold by qscore (preferred) OR legacy cls threshold
            keep = torch.zeros(Q, dtype=torch.bool, device=mask_logits.device)

            if topk is not None and int(topk) > 0:
                k = min(int(topk), Q)
                top_idx = torch.topk(qscore[b], k=k, largest=True).indices
                keep[top_idx] = True
            elif qscore_threshold is not None:
                keep = qscore[b] > float(qscore_threshold)
            else:
                # Legacy fallback (not train-aligned, but preserved)
                if cls_threshold is None:
                    cls_threshold = getattr(self, "default_cls_threshold", 0.5)
                keep = cls_probs[b] > float(cls_threshold)

            # Optional additional filter
            if min_mask_mass is not None:
                keep = keep & (mask_mass[b] > float(min_mask_mass))

            gate_pass = True
            if presence_threshold is not None:
                gate_pass = bool((presence_prob[b] > float(presence_threshold)).detach().cpu())
                if not gate_pass:
                    keep = torch.zeros(Q, dtype=torch.bool, device=mask_logits.device)

            num_keep = int(keep.sum().detach().cpu())

            if num_keep == 0:
                outputs.append({
                    "masks": torch.zeros((0, Hm, Wm), dtype=torch.uint8, device=mask_logits.device),
                    "mask_scores": torch.empty(0, device=mask_logits.device),
                    "mask_forgery_scores": torch.empty(0, device=mask_logits.device),
                    "image_authenticity": image_presence,   # prob forged (presence_prob)
                    # logging / debugging
                    "presence_prob": image_presence,
                    "gate_pass": gate_pass,
                    "max_cls_prob": max_cls_prob,
                    "max_qscore": max_qscore,
                    "num_keep": 0,
                    "max_mask_prob": max_mask_prob,
                    "mean_mask_mass": mean_mask_mass,
                    "max_mask_mass": max_mask_mass,
                    "any_fg_pre_keep": any_fg_pre_keep,
                    "any_fg_post_keep": False,
                    "selection_mode": "topk" if (topk is not None and int(topk) > 0)
                                    else ("qscore_thr" if qscore_threshold is not None else "cls_thr"),
                    "mask_threshold": float(mask_threshold),
                    "qscore_threshold": None if qscore_threshold is None else float(qscore_threshold),
                    "cls_threshold": None if cls_threshold is None else float(cls_threshold),
                    "topk": None if topk is None else int(topk),
                    "min_mask_mass": None if min_mask_mass is None else float(min_mask_mass),
                    "presence_threshold": None if presence_threshold is None else float(presence_threshold),
                })
                continue

            # Build kept outputs
            kept_mask_probs = mask_probs[b, keep]  # [K,H,W]
            masks_b = (kept_mask_probs > mask_threshold).to(torch.uint8)

            scores_b = qscore[b, keep]             # [K] qscore (train-aligned activity)
            cls_b = cls_probs[b, keep]             # [K] cls prob

            any_fg_post_keep = bool((kept_mask_probs > mask_threshold).any().detach().cpu())

            outputs.append({
                "masks": masks_b,
                "mask_scores": scores_b,
                "mask_forgery_scores": cls_b,
                "image_authenticity": image_presence,  # prob forged (presence_prob)
                # logging / debugging
                "presence_prob": image_presence,
                "gate_pass": gate_pass,
                "max_cls_prob": max_cls_prob,
                "max_qscore": max_qscore,
                "num_keep": num_keep,
                "max_mask_prob": max_mask_prob,
                "mean_mask_mass": mean_mask_mass,
                "max_mask_mass": max_mask_mass,
                "any_fg_pre_keep": any_fg_pre_keep,
                "any_fg_post_keep": any_fg_post_keep,
                "selection_mode": "topk" if (topk is not None and int(topk) > 0)
                                else ("qscore_thr" if qscore_threshold is not None else "cls_thr"),
                "mask_threshold": float(mask_threshold),
                "qscore_threshold": None if qscore_threshold is None else float(qscore_threshold),
                "cls_threshold": None if cls_threshold is None else float(cls_threshold),
                "topk": None if topk is None else int(topk),
                "min_mask_mass": None if min_mask_mass is None else float(min_mask_mass),
                "presence_threshold": None if presence_threshold is None else float(presence_threshold),
            })

        return outputs