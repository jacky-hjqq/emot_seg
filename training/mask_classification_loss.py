# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)
import torch.nn.functional as F


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        no_object_coefficient: float,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.eos_coef = no_object_coefficient

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

        # Mask-only matching (used for proposal supervision on all annotated instances).
        # We set class cost to 0 so matching depends only on mask/dice.
        self.matcher_mask = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=0.0,
        )

    @torch.compiler.disable
    def _match_mask_only(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
    ):
        """Mask-only Hungarian matching.

        Hugging Face's `Mask2FormerHungarianMatcher` API requires `class_queries_logits`
        and `class_labels` arguments even when `cost_class=0`. This wrapper hides that
        requirement from callers by supplying dummy tensors with the correct shapes.
        """

        # In this codebase, class logits are typically a per-sample list. Mirror that.
        B, Q = masks_queries_logits.shape[:2]
        device = masks_queries_logits.device

        dummy_class_queries_logits = [
            masks_queries_logits.new_zeros((Q, 1)) for _ in range(B)
        ]
        dummy_class_labels = [
            torch.zeros((m.shape[0],), dtype=torch.long, device=device) for m in mask_labels
        ]

        return self.matcher_mask(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=dummy_class_queries_logits,
            class_labels=dummy_class_labels,
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        # Conditioned targets (used for class supervision/matching to input conditions)
        mask_labels_cond = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels_cond = [target["labels"].long() for target in targets]

        # All-instance targets (used for mask proposal supervision). If absent, fall back to conditioned masks.
        mask_labels_all = [
            target.get("masks_all", target["masks"]).to(masks_queries_logits.dtype)
            for target in targets
        ]

        # 1) Mask proposal matching: match queries to ALL annotated instances (mask/dice only)
        indices_all = self._match_mask_only(masks_queries_logits, mask_labels_all)

        # 2) Condition matching: match queries to CONDITIONED instances using mask+dice+class cost.
        indices_cond = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels_cond,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels_cond,
        )

        # Supervise masks on all annotated instances.
        loss_masks = self.loss_masks(masks_queries_logits, mask_labels_all, indices_all)

        # Supervise class logits only for conditioned targets and background queries.
        loss_classes = self.loss_labels(
            class_queries_logits,
            class_labels_cond,
            indices_cond,
            indices_all=indices_all,
        )

        return {**loss_masks, **loss_classes}

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn, prefix="train") -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/{prefix}_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn(f"losses/{prefix}_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore
    
    def loss_labels(self, class_queries_logits, class_labels, indices, indices_all=None):
        """
        Compute the cross-entropy loss for labels.
        Modified to handle a variable number of classes per sample (List input).

        Args:
            class_queries_logits (`list[torch.Tensor]`):
                A list of length batch_size. Each element is a tensor of shape 
                `(num_queries, num_classes_i + 1)`.
            class_labels (`list[torch.Tensor]`):
                List of ground truth class labels.
            indices (`list[tuple[np.array, np.array]]`):
                List of tuples (src_idx, tgt_idx) computed by the Hungarian matcher.

        Returns:
            `dict[str, Tensor]`: A dict containing the cross entropy loss.
        """
        losses = []
        # Iterate over each sample in the batch individually
        if indices_all is None:
            indices_all = indices

        for sample_i, (logits, targets, (src_idx, tgt_idx), (src_idx_all, _tgt_idx_all)) in enumerate(
            zip(class_queries_logits, class_labels, indices, indices_all)
        ):
            # logits shape: (num_queries, num_classes_sample_i + 1)
            # The +1 accounts for the "No Object" class
            
            num_queries = logits.shape[0]
            num_classes_total = logits.shape[1] 
            
            # 1. Determine the background class index (last index)
            bg_idx = num_classes_total - 1
            
            # 2. Build Target Tensor
            # Initialize all targets as background first
            target_classes = torch.full(
                (num_queries,), 
                fill_value=bg_idx, 
                dtype=torch.int64, 
                device=logits.device
            )
            
            # Assign ground truth labels to the matched queries
            # src_idx: Indices of the predicted queries
            # tgt_idx: Indices of the corresponding ground truth labels
            if len(src_idx) > 0:
                src_idx_t = torch.as_tensor(src_idx, dtype=torch.long, device=logits.device)
                tgt_idx_t = torch.as_tensor(tgt_idx, dtype=torch.long, device=logits.device)
                target_classes[src_idx_t] = targets[tgt_idx_t]
            else:
                src_idx_t = torch.empty((0,), dtype=torch.long, device=logits.device)
            
            # 3. Dynamic Weight Tensor
            # Since the number of classes varies per sample, we construct the weight tensor dynamically.
            # Foreground weights = 1.0, Background weight = eos_weight
            weights = torch.ones(num_classes_total, device=logits.device)
            weights[-1] = self.eos_coef

            # 4. Compute per-query CE and selectively ignore queries that were matched
            # to unconditioned objects (present in indices_all but not in indices_cond).
            per_query_loss = F.cross_entropy(
                logits,
                target_classes,
                weight=weights,
                reduction="none",
            )

            src_idx_all_t = (
                torch.as_tensor(src_idx_all, dtype=torch.long, device=logits.device)
                if len(src_idx_all) > 0
                else torch.empty((0,), dtype=torch.long, device=logits.device)
            )

            supervise = torch.ones((num_queries,), dtype=torch.bool, device=logits.device)
            # Exclude queries assigned to ANY (all-instance) object...
            if src_idx_all_t.numel() > 0:
                supervise[src_idx_all_t] = False
            # ...but keep supervision for the conditioned matches.
            if src_idx_t.numel() > 0:
                supervise[src_idx_t] = True

            if supervise.any():
                loss_sample = per_query_loss[supervise].mean()
            else:
                loss_sample = per_query_loss.new_zeros(())

            losses.append(loss_sample)

        # 5. Aggregate losses (Stack and Mean over the batch)
        final_loss = torch.stack(losses).mean()
        losses = {"loss_cross_entropy": final_loss}

        return losses
