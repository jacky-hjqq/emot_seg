# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import torch
import lightning


class LightningDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        path,
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        check_empty_targets: bool,
        ignore_idx: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()

        self.path = path
        self.check_empty_targets = check_empty_targets
        self.ignore_idx = ignore_idx
        self.img_size = img_size

        self.dataloader_kwargs = {
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "batch_size": batch_size,
        }

    @staticmethod
    def train_collate(batch):
        imgs, list_targets, bs_cond_imgs, bs_cond_masks = [], [], [], []

        for data in batch:
            img = data["img"] # (3,h,w)
            targets = data["targets"] 
            cond_imgs = data["cond_imgs"] # (n,3,h_c,w_c)
            cond_masks = data["cond_masks"] # (n,h_c,w_c)
            imgs.append(img)
            list_targets.append(targets)
            bs_cond_imgs.append(cond_imgs)
            bs_cond_masks.append(cond_masks)

        return torch.stack(imgs), list_targets, torch.stack(bs_cond_imgs), torch.stack(bs_cond_masks)

    @staticmethod
    def eval_collate(batch):
        imgs, list_targets, bs_cond_imgs, bs_cond_masks = [], [], [], []

        for data in batch:
            img = data["img"] # (3,h,w)
            targets = data["targets"]
            cond_imgs = data["cond_imgs"] 
            cond_masks = data["cond_masks"]
            imgs.append(img)
            list_targets.append(targets)
            bs_cond_imgs.append(cond_imgs)
            bs_cond_masks.append(cond_masks)

        return torch.stack(imgs), list_targets, torch.stack(bs_cond_imgs), torch.stack(bs_cond_masks)