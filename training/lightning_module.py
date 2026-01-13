# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the torchmetrics library by the PyTorch Lightning team
# - the Mask2Former repository by Facebook, Inc. and its affiliates
# All used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
from typing import Optional
import lightning
import torch
import torch.nn as nn
from torch.optim import AdamW
import wandb
from PIL import Image
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms.v2.functional import pad
import logging
import torch.nn.functional as F

from training.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule

from datasets.utils.vis_utils import visualize_targets_and_templates, visualize_image_mask_and_templates

bold_green = "\033[1;32m"
reset = "\033[0m"


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]],
        attn_mask_annealing_end_steps: Optional[list[int]],
        lr: float,
        llrd: float,
        llrd_l2_enabled: bool,
        lr_mult: float,
        weight_decay: float,
        poly_power: float,
        warmup_steps: tuple[int, int],
        ckpt_path=None,
        delta_weights=False,
        load_ckpt_class_head=True,
        train_vis_interval: Optional[int] = None,
        val_vis_interval: Optional[int] = None,
    ):
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps
        self.lr = lr
        self.llrd = llrd
        self.lr_mult = lr_mult
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps
        self.llrd_l2_enabled = llrd_l2_enabled
        self.train_vis_interval = train_vis_interval
        self.val_vis_interval = val_vis_interval

        self.strict_loading = False

        if delta_weights and ckpt_path:
            logging.info("Delta weights mode")
            self._zero_init_outside_encoder(skip_class_head=not load_ckpt_class_head)
            current_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
            if not load_ckpt_class_head:
                current_state_dict = {
                    k: v
                    for k, v in current_state_dict.items()
                    if "class_head" not in k and "class_predictor" not in k
                }
            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            combined_state_dict = self._add_state_dicts(current_state_dict, ckpt)
            incompatible_keys = self.load_state_dict(combined_state_dict, strict=False)
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)
        elif ckpt_path:
            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            incompatible_keys = self.load_state_dict(ckpt, strict=False)
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)

        self.log = torch.compiler.disable(self.log)  # type: ignore

    def configure_optimizers(self):
        encoder_param_names = {
            n for n, _ in self.network.encoder.backbone.named_parameters()
        }
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = len(self.network.encoder.backbone.blocks)
        block_i = backbone_blocks

        l2_blocks = torch.arange(
            backbone_blocks - self.network.num_blocks, backbone_blocks
        ).tolist()

        for name, param in reversed(list(self.named_parameters())):
            lr = self.lr

            if name.replace("network.encoder.backbone.", "") in encoder_param_names:
                name_list = name.split(".")

                is_block = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True

                if is_block or block_i == 0:
                    lr *= self.llrd ** (backbone_blocks - 1 - block_i)

                elif (is_block or block_i == 0) and self.lr_mult != 1.0:
                    lr *= self.lr_mult

                if "backbone.norm" in name:
                    lr = self.lr

                if (
                    is_block
                    and (block_i in l2_blocks)
                    and ((not self.llrd_l2_enabled) or (self.lr_mult != 1.0))
                ):
                    lr = self.lr

                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": self.lr, "name": name}
                )

        param_groups = backbone_param_groups + other_param_groups
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, imgs, bs_cond_imgs, bs_cond_masks):
        imgs = imgs / 255.0
        bs_cond_imgs = bs_cond_imgs / 255.0

        return self.network(imgs, bs_cond_imgs, bs_cond_masks)

    def training_step(self, batch, batch_idx):
        imgs, list_targets, bs_cond_imgs, bs_cond_masks = batch

        mask_logits_per_block, class_logits_per_block = self(imgs, bs_cond_imgs, bs_cond_masks)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=list_targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses
        
        # Visualization every N steps 
        if (
            hasattr(self, 'train_vis_interval')
            and self.train_vis_interval is not None
            and self.train_vis_interval > 0
            and (self.global_step % self.train_vis_interval == 0)
        ):
            img_sizes = [img.shape[-2:] for img in imgs]
            # use the last block for visualization and only visualize the first batch item
            # scale up to original image size
            vis_mask_logits = F.interpolate(mask_logits_per_block[-1], self.img_size, mode="bilinear")
            vis_mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                vis_mask_logits, img_sizes
            )
            for b in range(len(imgs)):
                pred = self.to_per_pixel_preds_instance(
                    vis_mask_logits[b],
                    class_logits_per_block[-1][b],
                    self.mask_thresh,
                    self.overlap_thresh,
                )
                semantic_mask, instance_mask = pred[..., 0], pred[..., 1]

                pred_vis = visualize_image_mask_and_templates(
                    img=imgs[b], 
                    mask=semantic_mask,
                    cond_imgs=bs_cond_imgs[b]
                )
                gt_vis = visualize_targets_and_templates(
                    img=imgs[b], 
                    targets=list_targets[b],
                    cond_imgs=bs_cond_imgs[b]
                )
                break

            # Log pred_vis and gt_vis to wandb
            if self.trainer.logger is not None:
                self.trainer.logger.experiment.log({
                    "train/pred": wandb.Image(pred_vis),
                    "train/gt": wandb.Image(gt_vis)
                })

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    def mask_annealing(self, start_iter, current_iter, final_iter):
        device = self.device
        dtype = self.network.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(self.poly_power)

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        if self.attn_mask_annealing_enabled:
            for i in range(self.network.num_blocks):
                self.network.attn_mask_probs[i] = self.mask_annealing(
                    self.attn_mask_annealing_start_steps[i],
                    self.global_step,
                    self.attn_mask_annealing_end_steps[i],
                )

            for i, attn_mask_prob in enumerate(self.network.attn_mask_probs):
                self.log(
                    f"attn_mask_prob_{i}",
                    attn_mask_prob,
                    on_step=True,
                )

    def block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-5 + block_idx + 1}" # 5 means number of layers with masked attn
            if block_idx != self.network.num_blocks
            else ""
        )

    @torch.compiler.disable
    def scale_img_size_semantic(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def window_imgs_semantic(self, imgs):
        crops, origins = [], []

        for i in range(len(imgs)):
            img = imgs[i]
            new_h, new_w = self.scale_img_size_semantic(img.shape[-2:])
            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).to(img.device)
            )

            num_crops = math.ceil(max(resized_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(resized_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                if resized_img.shape[-2] > resized_img.shape[-1]:
                    crop = resized_img[:, start:end, :]
                else:
                    crop = resized_img[:, :, start:end]

                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins

    def revert_window_logits_semantic(self, crop_logits, origins, img_sizes):
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate(
                (sums / counts)[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        targets: list[dict],
        ignore_idx,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def scale_img_size_instance_panoptic(self, size: tuple[int, int]):
        factor = min(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def resize_and_pad_imgs_instance_panoptic(self, imgs):
        transformed_imgs = []

        for img in imgs:
            new_h, new_w = self.scale_img_size_instance_panoptic(img.shape[-2:])

            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(img.device)
            )

            pad_h = max(0, self.img_size[-2] - resized_img.shape[-2])
            pad_w = max(0, self.img_size[-1] - resized_img.shape[-1])
            padding = [0, 0, pad_w, pad_h]

            padded_img = pad(resized_img, padding)

            transformed_imgs.append(padded_img)

        return torch.stack(transformed_imgs)

    @torch.compiler.disable
    def revert_resize_and_pad_logits_instance_panoptic(
        self, transformed_logits, img_sizes
    ):
        logits = []
        for i in range(len(transformed_logits)):
            scaled_size = self.scale_img_size_instance_panoptic(img_sizes[i])
            logits_i = transformed_logits[i][:, : scaled_size[0], : scaled_size[1]]
            logits_i = interpolate(
                logits_i[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            logits.append(logits_i)

        return logits
    
    @torch.compiler.disable
    def to_per_pixel_preds_instance(
        self,
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
        mask_thresh: float,
        overlap_thresh: float,
        return_scores: bool = False,
    ):
        """
        Post-processing for dynamic/conditional instance segmentation.
        Class IDs correspond to the indices of condition images.
        
        Args:
            mask_logits: Shape (Q, H, W) - Mask predictions for a single sample.
            class_logits: Shape (Q, Num_Cond_Imgs + 1). 
                          Assumes the LAST class index is "No Object" (background).
            mask_thresh: Score threshold.
            overlap_thresh: Occlusion pruning threshold.

        Returns:
            If return_scores is False:
                preds: Shape (H, W, 2).
                    Layer 0: Condition Image Index (Class ID). -1 means Background.
                    Layer 1: Instance ID. -1 means Background.

            If return_scores is True:
                (preds, instance_scores)
                instance_scores: Shape (N,). Score per returned instance ID, where
                    instance_scores[i] corresponds to instance_id == i in preds[..., 1].
        """
        # 1. Get scores and classes
        # Dynamic: The number of classes depends on the input shape
        probs = class_logits.softmax(dim=-1)
        topk = min(2, probs.shape[-1])
        scores_k, classes_k = probs.topk(k=topk, dim=-1)  # (Q, K)
        scores = scores_k.reshape(-1)  # (Q*K,)
        classes = classes_k.reshape(-1)  # (Q*K,)
        
        # Dynamically determine the "No Object" class index
        # Usually the last dimension of logits contains the N classes + 1 void class
        no_object_idx = class_logits.shape[-1] - 1
        
        # Initialize the output map: (H, W, 2)
        # Fill with -1. This effectively sets Background = -1
        preds = -torch.ones(
            (*mask_logits.shape[-2:], 2),
            dtype=torch.long,
            device=class_logits.device,
        )

        # 2. Filtering
        # Keep if class is NOT "No Object" AND score > threshold
        keep = classes.ne(no_object_idx) & (scores > mask_thresh)
        
        if not keep.any():
            if return_scores:
                empty_scores = torch.empty(
                    (0,), dtype=scores.dtype, device=class_logits.device
                )
                return preds, empty_scores
            return preds

        # 3. Pixel-wise Competition
        masks = mask_logits.sigmoid()
        if topk > 1:
            masks = masks.repeat_interleave(topk, dim=0)  # (Q*K, H, W)
        mask_ids = (scores[keep][..., None, None] * masks[keep]).argmax(0)
        
        segments = -torch.ones(
            *masks.shape[-2:],
            dtype=torch.long,
            device=class_logits.device,
        )

        segment_id = 0
        segment_and_class_ids = []
        instance_scores_list = []

        for k, class_id in enumerate(classes[keep].tolist()):
            orig_mask = masks[keep][k] >= 0.5
            new_mask = mask_ids == k
            final_mask = orig_mask & new_mask

            orig_area = orig_mask.sum().item()
            new_area = new_mask.sum().item()
            final_area = final_mask.sum().item()

            if (
                orig_area == 0
                or new_area == 0
                or final_area == 0
                or new_area / orig_area < overlap_thresh
            ):
                continue

            segments[final_mask] = segment_id
            segment_and_class_ids.append((segment_id, class_id))
            instance_scores_list.append(scores[keep][k])
            
            segment_id += 1

        # 4. Final Map Construction
        for seg_id, cls_id in segment_and_class_ids:
            segment_mask = segments == seg_id
            
            # Assign the Condition Image Index as the Class ID
            preds[:, :, 0] = torch.where(segment_mask, cls_id, preds[:, :, 0])
            # Assign the unique Instance ID
            preds[:, :, 1] = torch.where(segment_mask, seg_id, preds[:, :, 1])

        if return_scores:
            if instance_scores_list:
                instance_scores = torch.stack(instance_scores_list)
            else:
                instance_scores = torch.empty(
                    (0,), dtype=scores.dtype, device=class_logits.device
                )
            return preds, instance_scores

        return preds

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    def _zero_init_outside_encoder(
        self, encoder_prefix="network.encoder.", skip_class_head=False
    ):
        with torch.no_grad():
            total, zeroed = 0, 0
            for name, p in self.named_parameters():
                total += p.numel()
                if not name.startswith(encoder_prefix):
                    if skip_class_head and (
                        "class_head" in name or "class_predictor" in name
                    ):
                        continue
                    p.zero_()
                    zeroed += p.numel()
            msg = f"Zeroed {zeroed:,} / {total:,} parameters (everything not under '{encoder_prefix}'"
            if skip_class_head:
                msg += ", skipping class head"
            msg += ")"
            logging.info(msg)

    def _add_state_dicts(self, state_dict1, state_dict2):
        summed = {}
        for k in state_dict1.keys():
            if k not in state_dict2:
                raise KeyError(f"Key {k} not found in second state_dict")

            if state_dict1[k].shape != state_dict2[k].shape:
                raise ValueError(
                    f"Shape mismatch at {k}: "
                    f"{state_dict1[k].shape} vs {state_dict2[k].shape}"
                )

            summed[k] = state_dict1[k] + state_dict2[k]

        return summed

    def _load_ckpt(self, ckpt_path, load_ckpt_class_head):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}
        if not load_ckpt_class_head:
            ckpt = {
                k: v
                for k, v in ckpt.items()
                if "class_head" not in k and "class_predictor" not in k
            }
        logging.info(f"Loaded {len(ckpt)} keys")
        return ckpt

    def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head):
        if incompatible_keys.missing_keys:
            if not load_ckpt_class_head:
                missing_keys = [
                    key
                    for key in incompatible_keys.missing_keys
                    if "class_head" not in key and "class_predictor" not in key
                ]
            else:
                missing_keys = incompatible_keys.missing_keys
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")
        if incompatible_keys.unexpected_keys:
            raise ValueError(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
