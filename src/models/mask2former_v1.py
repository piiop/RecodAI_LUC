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

class Mask2FormerForgeryModel(nn.Module):
    """
    Instance-Seg Transformer (Mask2Former-style) + Authenticity Gate baseline.
    """

    @staticmethod
    def _coerce_thresh(name, value, default=0.5):
        if value is None:
            print(f"[Warning] `{name}` was None — coercing to default={default}.")
            return default
        return value

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
        auth_gate_forged_threshold=0.0,
        default_mask_threshold=0.0,
        default_cls_threshold=0.0,
        # training thresholds
        auth_penalty_cls_threshold=None,
        auth_penalty_temperature=0.1,
        # matching weights
        cost_bce=1.0,
        cost_dice=1.0,
        # loss weights (for convenience total loss)
        authenticity_penalty_weight=5.0,        
        loss_weight_mask_bce=1.0,
        loss_weight_mask_dice=1.0,
        loss_weight_mask_cls=1.0,
        loss_weight_img_auth=1.0,
        loss_weight_auth_penalty=1.0,
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
        # Thresholds (validated / coerced)
        # -----------------------------
        self.auth_gate_forged_threshold = auth_gate_forged_threshold
        self.default_mask_threshold = default_mask_threshold
        self.default_cls_threshold = self._coerce_thresh(
            "default_cls_threshold", default_cls_threshold
        )
        self.auth_penalty_cls_threshold = self._coerce_thresh(
            "auth_penalty_cls_threshold",
            auth_penalty_cls_threshold
            if auth_penalty_cls_threshold is not None
            else default_cls_threshold,
        )
        self.auth_penalty_temperature = auth_penalty_temperature

        # -----------------------------
        # Matching weights
        # -----------------------------
        self.cost_bce = cost_bce
        self.cost_dice = cost_dice

        # -----------------------------
        # Loss weights
        # -----------------------------
        self.loss_weight_mask_bce = loss_weight_mask_bce
        self.loss_weight_mask_dice = loss_weight_mask_dice
        self.loss_weight_mask_cls = loss_weight_mask_cls
        self.loss_weight_img_auth = loss_weight_img_auth
        self.loss_weight_auth_penalty = loss_weight_auth_penalty

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

        # -----------------------------
        # Image-level authenticity head
        # -----------------------------
        self.img_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )


    def forward(self, images, targets=None, inference_overrides=None):
        """
        images: Tensor [B, 3, H, W] or list[Tensor(C,H,W)]
        targets: list[dict] or None
        """
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        B = images.shape[0]

        fpn_feats = self.backbone(images)  # [P2, P3, P4, P5]

        # Mask2Former pixel decoder produces:
        # - memory_feats: [P3,P4,P5] (for transformer decoder)
        # - mask_feats: high-res mask features at P2 resolution
        memory_feats, mask_feats = self.pixel_decoder(fpn_feats, self.position_encoding)

        pos_list = [self.position_encoding(x) for x in memory_feats]
        hs_all = self.transformer_decoder(memory_feats, pos_list)  # [num_layers, B, Q, C]

        hs = hs_all[-1]  # [B, Q, C]

        # Heads
        class_logits = self.class_head(hs).squeeze(-1)     # [B, Q]
        mask_embeddings = self.mask_embed_head(hs)         # [B, Q, mask_dim]
        mask_logits = torch.einsum("bqc, bchw -> bqhw", mask_embeddings, mask_feats)

        # Image-level authenticity
        high_level_feat = fpn_feats[-1]
        img_logits = self.img_head(high_level_feat).squeeze(-1)  # [B]

        # ---- training loss ----
        if self.training and targets is not None:
            overrides = inference_overrides or {}
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
                auth_penalty_temperature=self.auth_penalty_temperature,
                logger=overrides.get("logger", None),
                debug_ctx=overrides.get("debug_ctx", None),
            )

        # ---- inference (also used when targets is None) ----
        overrides = inference_overrides or {}
        return self.inference(
            mask_logits=mask_logits,
            class_logits=class_logits,
            img_logits=img_logits,
            mask_threshold=overrides.get("mask_threshold", None),
            cls_threshold=overrides.get("cls_threshold", None),
            auth_gate_forged_threshold=overrides.get("auth_gate_forged_threshold", None),
        )

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
        img_logits = self.img_head(fpn_feats[-1]).squeeze(-1)  # [B]
        return mask_logits, class_logits, img_logits

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
            # TEMPORARY DIAGNOSTIC: decouple inference from image gate entirely
            gate_pass = True
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
