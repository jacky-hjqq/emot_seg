# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.scale_block import ScaleBlock
import numpy as np


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        cond_encoder: nn.Module,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.cond_encoder = cond_encoder
        # Forzen cond encoder
        for param in self.cond_encoder.parameters():
            param.requires_grad = False

        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)
        
        # Adapter to project cond_encoder features to encoder embedding dimension
        input_dim = self.cond_encoder.backbone.embed_dim
        output_dim = self.encoder.backbone.embed_dim
        self.cond_adapter = Adapter(
            in_features=input_dim,
            out_features=output_dim,
            reduction=4,  # Standard reduction factor
            ratio=0.2     # Standard residual ratio (alpha)
        )
        # ---------------------------------------
        self.class_cross_attn = nn.MultiheadAttention(
            embed_dim=self.encoder.backbone.embed_dim,
            num_heads=8,
            batch_first=True
        )
        # Optional: learnable temperature for scaling logits
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # ---------------------------------------

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )

    def _predict(self, x: torch.Tensor, cond_feats: torch.Tensor):
        q = x[:, : self.num_q, :]

        # Apply adapter to condition features
        cond_feats_adapted = self.cond_adapter(cond_feats)
        
        # class_logits = self.class_head(q)
        attended_features, attn_weights = self.class_cross_attn(
            query=q,
            key=cond_feats_adapted,
            value=cond_feats_adapted,
            need_weights=True,
            average_attn_weights=True
        )
        # Scale the logits (learnable temperature)
        class_logits = attn_weights * self.logit_scale.exp()
        
        # Add "no object" class
        no_object_logits = torch.zeros(
            class_logits.shape[0], 
            class_logits.shape[1], 
            1, 
            device=class_logits.device
        )
        class_logits = torch.cat([class_logits, no_object_logits], dim=-1)
        class_logits = class_logits.squeeze(0) # remove the bs dim

        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(
            mask_logits,
            self.encoder.backbone.patch_embed.grid_size,
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[
            :,
            : self.num_q,
            self.num_q + self.encoder.backbone.num_prefix_tokens :,
        ] = (
            interpolated > 0
        )
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[
                i - len(self.encoder.backbone.blocks) + self.num_blocks
            ],
        )
        return attn_mask

    def forward(self, x: torch.Tensor, list_cond_x: torch.Tensor = None, list_cond_masks: torch.Tensor = None):
        # Extract condition features from all layers for each sample in the batch
        if list_cond_x is not None:
            batch_size = x.shape[0]
            list_cond_feats = []
            for b in range(batch_size):
                cond_x = (list_cond_x[b] - self.cond_encoder.pixel_mean) / self.cond_encoder.pixel_std
                cond_mask = list_cond_masks[b] if list_cond_masks is not None else None
                cond_layer_feats = self.encode_cond_x_multilayer(cond_x, cond_mask)
                # List of 5 tensors: (num_cond_b, embed_dim) where num_cond_b varies per batch
                list_cond_feats.append(cond_layer_feats) 
        else:
            raise ValueError("cond_x is required for open-vocabulary mode")

        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []
        layer_idx = 0  # Track which cond features to use

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                # Process each sample in the batch separately due to varying num_cond
                batch_mask_logits = []
                batch_class_logits = []
                
                for b in range(batch_size):
                    x_b = x[b:b+1]  # (1, num_tokens, embed_dim)
                    cond_feats_b = list_cond_feats[b][layer_idx]  # (num_cond_b, embed_dim)
                    
                    mask_logits_b, class_logits_b = self._predict(
                        self.encoder.backbone.norm(x_b),
                        cond_feats_b.unsqueeze(0)  # (1, num_cond_b, embed_dim)
                    )
                    batch_mask_logits.append(mask_logits_b)
                    batch_class_logits.append(class_logits_b)
                
                mask_logits = torch.cat(batch_mask_logits, dim=0)
                class_logits = batch_class_logits  # Keep as list due to varying num_cond
                
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                layer_idx += 1

                attn_mask = self._attn_mask(x, mask_logits, i)

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)

        # Final prediction with final layer's cond features
        batch_mask_logits = []
        batch_class_logits = []
        
        for b in range(batch_size):
            x_b = x[b:b+1]  # (1, num_tokens, embed_dim)
            cond_feats_b = list_cond_feats[b][layer_idx]  # (num_cond_b, embed_dim)
            
            mask_logits_b, class_logits_b = self._predict(
                self.encoder.backbone.norm(x_b),
                cond_feats_b.unsqueeze(0)  # (1, num_cond_b, embed_dim)
            )
            batch_mask_logits.append(mask_logits_b)
            batch_class_logits.append(class_logits_b)
        
        mask_logits = torch.cat(batch_mask_logits, dim=0)
        class_logits = batch_class_logits  # Keep as list due to varying num_cond
        
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
    
    def encode_cond_x_multilayer(self, cond_x: torch.Tensor, cond_masks: torch.Tensor = None):
        """
        Extract features from multiple layers for condition images.
        Each cond_x is treated as a separate class/instance.
        
        Args:
            cond_x: (num_cond, C, H, W) - Input images. No Batch dimension.
            cond_masks: (num_cond, H, W) - Binary mask (1=valid, 0=masked). Optional.
        
        Returns: 
            List of (num_cond, embed_dim), one tensor per extracted layer.
        """
        num_cond, C, H, W = cond_x.shape
        
        # Prepare masks for patch-level masking
        if cond_masks is not None:            
            # Downsample masks to match the patch grid size of the ViT backbone
            patch_h, patch_w = self.cond_encoder.backbone.patch_embed.grid_size
            masks_patches = F.interpolate(
                cond_masks.unsqueeze(1).float(),
                size=(patch_h, patch_w),
                mode='nearest'
            ).squeeze(1)  # (num_cond, patch_h, patch_w)
            
            # Flatten spatial dimensions to match sequence length: (num_cond, num_patches)
            masks_patches = masks_patches.view(num_cond, -1) 
        else:
            masks_patches = None
        
        # Encode through backbone
        # The backbone expects (Batch, C, H, W), here we use num_cond as Batch
        x = self.cond_encoder.backbone.patch_embed(cond_x)
        
        if hasattr(self.cond_encoder.backbone, "_pos_embed"):
            x = self.cond_encoder.backbone._pos_embed(x)
        
        # Determine number of prefix tokens (e.g., CLS token)
        num_prefix_tokens = self.cond_encoder.backbone.num_prefix_tokens if hasattr(self.cond_encoder.backbone, 'num_prefix_tokens') else 0
        
        cond_features_per_layer = []
        
        for i, block in enumerate(self.cond_encoder.backbone.blocks):
            x = block(x)
            
            # Extract features from the last `self.num_blocks` layers
            if i >= len(self.cond_encoder.backbone.blocks) - self.num_blocks:
                normed = self.cond_encoder.backbone.norm(x)
                
                # Apply Masked Global Average Pooling
                if masks_patches is not None:
                    # normed: (num_cond, num_prefix_tokens + num_patches, embed_dim)
                    # Extract only patch tokens (skip prefix tokens like CLS)
                    patch_tokens = normed[:, num_prefix_tokens:, :]  # (num_cond, num_patches, embed_dim)
                    
                    # masks_patches: (num_cond, num_patches)
                    mask_expanded = masks_patches.unsqueeze(-1)
                    masked_features = patch_tokens * mask_expanded
                    
                    # Sum features and divide by the number of valid patches (weighted average)
                    features = masked_features.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
                else:
                    # Standard Global Average Pooling (skip prefix tokens)
                    patch_tokens = normed[:, num_prefix_tokens:, :]
                    features = patch_tokens.mean(dim=1)  # (num_cond, embed_dim)
                
                cond_features_per_layer.append(features)
        
        # Extract features from the final output (usually after the final norm block of ViT)
        normed = self.cond_encoder.backbone.norm(x)
        
        if masks_patches is not None:
            patch_tokens = normed[:, num_prefix_tokens:, :]
            mask_expanded = masks_patches.unsqueeze(-1)
            masked_features = patch_tokens * mask_expanded
            features = masked_features.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            patch_tokens = normed[:, num_prefix_tokens:, :]
            features = patch_tokens.mean(dim=1)
            
        cond_features_per_layer.append(features)
        
        return cond_features_per_layer

class Adapter(nn.Module):
    def __init__(self, in_features, out_features, reduction=4, ratio=0.2):
        """
        Args:
            in_features: Input dimension (cond_encoder output)
            out_features: Output dimension (encoder input)
            reduction: Bottleneck reduction factor (default 4 per original paper)
            ratio: Residual mixing ratio (alpha). Default 0.2 (20% adapter, 80% original)
        """
        super().__init__()
        self.ratio = ratio
        
        # 1. Dimension Alignment (Extra step needed if in_dim != out_dim)
        if in_features != out_features:
            self.pre_project = nn.Linear(in_features, out_features, bias=False)
        else:
            self.pre_project = nn.Identity()

        # 2. The Adapter 
        hidden_dim = out_features // reduction
        self.adapter = nn.Sequential(
            nn.Linear(out_features, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        # 1. Align dimensions first
        x_raw = self.pre_project(x)
        
        # 2. Pass through bottleneck adapter
        x_adapted = self.adapter(x_raw)
        
        # 3. Weighted Residual Connection
        # formula: ratio * adapter(x) + (1 - ratio) * x
        out = self.ratio * x_adapted + (1 - self.ratio) * x_raw
        
        return out
