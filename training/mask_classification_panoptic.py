# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule

import wandb
from datasets.utils.vis_utils import visualize_targets_and_templates, visualize_image_mask_and_templates


class MaskClassificationPanoptic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        train_vis_interval: Optional[int] = None,
        val_vis_interval: Optional[int] = None,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            train_vis_interval=train_vis_interval,
            val_vis_interval=val_vis_interval,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            no_object_coefficient=no_object_coefficient,
        )

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, list_targets, bs_cond_imgs, bs_cond_masks = batch
        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs, bs_cond_imgs, bs_cond_masks)

        val_losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            val_losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=list_targets,
            )
            block_postfix = self.block_postfix(i)
            val_losses = {f"{key}{block_postfix}": value for key, value in val_losses.items()}
            val_losses_all_blocks |= val_losses

            # # Visualization
            # mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            # mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
            #     mask_logits, img_sizes
            # )
            # preds = []
            # for b in range(len(mask_logits)):
            #     assert len(mask_logits) == len(class_logits)
            #     pred = self.to_per_pixel_preds_instance(
            #         mask_logits[b],
            #         class_logits[b],
            #         self.mask_thresh,
            #         self.overlap_thresh,
            #     )
            #     preds.append(pred)

        # Visualization every N steps 
        if (
            hasattr(self, 'val_vis_interval')
            and self.val_vis_interval is not None
            and self.val_vis_interval > 0
            and (batch_idx % self.val_vis_interval == 0)
        ):
            img_sizes = [img.shape[-2:] for img in imgs]
            # use the last block for visualization and only visualize the first batch item
            # scale up to original image size
            vis_mask_logits = F.interpolate(mask_logits_per_layer[-1], self.img_size, mode="bilinear")
            vis_mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                vis_mask_logits, img_sizes
            )
            for b in range(len(imgs)):
                pred = self.to_per_pixel_preds_instance(
                    vis_mask_logits[b],
                    class_logits_per_layer[-1][b],
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
                    "val/pred": wandb.Image(pred_vis),
                    "val/gt": wandb.Image(gt_vis)
                })
        
        return self.criterion.loss_total(val_losses_all_blocks, self.log, prefix="val")

        
    def on_validation_epoch_end(self):
        pass

    def on_validation_end(self):
        pass
