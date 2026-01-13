import os
import json
import logging
import numpy as np
import cv2
from PIL import Image
import glob
import torch
import torch.utils.data as data
from collections import defaultdict
import random
import sys
sys.path.append(".")
from datasets.utils.augmentation import (
    get_image_augmentation,
    random_rotation,
    random_scaled_crop
)
from datasets.utils.img_utils import crop_and_resize, get_mask
from datasets.utils.vis_utils import *

class LMO_Train_Segmentation(data.Dataset):
    def __init__(self, LMODataset, augs=None):
        self.data_root = LMODataset["data_root"]
        self.cond_root = LMODataset['cond_root']
        self.condition_size = LMODataset["condition_size"]
        self.obj_ids = list(range(1, 16))

        # --- Augmentation Settings ---
        # Random crop settings
        self.random_crop = augs['random_crop']
        self.random_crop_ratio = augs['random_crop_ratio']
        self.random_crop_scale = augs['random_crop_scale']
        # Controls whether to apply identical color jittering
        self.cojitter = augs['cojitter']
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = augs['cojitter_ratio']
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            color_jitter=augs['color_jitter'],
            gray_scale=augs['gray_scale'],
            gau_blur=augs['gau_blur'],
        )
        
        self.total_data = []
        self.annotation = defaultdict(list)
        scenes = sorted(os.listdir(os.path.join(self.data_root)))
        # scenes = scenes[:1]
        for scene in scenes:
            scene_rgb_dir = os.path.join(self.data_root, scene, "rgb")
            frames = sorted([f for f in os.listdir(scene_rgb_dir) if f.endswith('.jpg')])
            scene_gt_path = os.path.join(self.data_root, scene, "scene_gt.json")
            with open(scene_gt_path, 'r') as f: 
                scene_gt = json.load(f)
            for frame in frames:
                frame_id = int(frame.split('.')[0])
                frame_gt = scene_gt[str(frame_id)]
                frame_image_path = os.path.join(scene_rgb_dir, frame)
                instance_info = []
                for inst_id, item in enumerate(frame_gt):
                    obj_id = item['obj_id']
                    obj_mask_path = os.path.join(self.data_root, scene, "mask_visib", f"{frame_id:06d}_{inst_id:06d}.png")
                    instance_info.append({
                        "obj_id": obj_id,
                        "mask_path": obj_mask_path,
                    })
                self.total_data.append({
                    "image_path": frame_image_path,
                    "instance_info": instance_info, 
                    "scene_id": int(scene),
                    "image_id": frame_id,
                    })

        self.len_val = len(self.total_data)                 
        logging.info(f"LMO Data size: {self.len_val}")
    
    def get_cond_data(self, obj_id):
        """Load a condition image/mask pair for the given ``obj_id``.

        - Prefer precomputed condition images under ``self.cond_root/<obj_id>/`` with files
          named ``<stem>_image.png`` and ``<stem>_mask.png`` (randomly choose one).
        - If no precomputed files are available, randomly sample one annotation instance from
          ``self.annotation_objects[obj_id]`` and extract the instance mask from that frame.
        Returns ``(cond_img, cond_mask)`` or ``(None, None)`` if no valid source exists.
        """
        obj_dir = os.path.join(self.cond_root, f"obj_{obj_id:06d}")
        # find all candidate image files that follow the naming convention
        img_files = [f for f in os.listdir(obj_dir) if f.endswith('_image.png')]
        if len(img_files) > 0:
            img_file = random.choice(img_files)
            stem = img_file[:-len('_image.png')]
            img_path = os.path.join(obj_dir, img_file)
            mask_path = os.path.join(obj_dir, stem + '_mask.png')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                cond_img = cv2.imread(img_path)[:, :, ::-1]
                cond_mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                # support single-channel or multi-channel masks
                if cond_mask_raw is not None:
                    if cond_mask_raw.ndim == 3:
                        cond_mask = cond_mask_raw[:, :, 0]
                    else:
                        cond_mask = cond_mask_raw
                    # augmentation
                    cond_img, cond_mask = self._cond_img_augmentation(cond_img, cond_mask)
                    # binarize: consider non-zero as foreground and accept as-is
                    cond_mask = (cond_mask != 0).astype(np.uint8) * 255
                    cond_img = cond_img * cond_mask[:, :, None].astype(bool)
                    return cond_img, cond_mask
        else:
            print(f"No precomputed cond images for obj {obj_id}, sampling from dataset.")
            return None, None
    
    def _img_augmentation(self, img, mask, mask_all):
        # Kept for backward compatibility with older code paths.
        # Prefer doing synchronized augmentation in __getitem__ when multiple masks are involved.
        # Random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask_all = mask_all.transpose(Image.FLIP_LEFT_RIGHT)
        # Apply Color Augmentation (training mode only)
        if self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                img = self.image_aug(img)

        return img, self._mask_transform(mask), self._mask_transform(mask_all)
    
    def _cond_img_augmentation(self, img, mask):
        # random rotation
        angle = random.uniform(-180, 180)
        aug_img, aug_mask = random_rotation(img, mask, angle, expand=True)
        # vis = visualize_cond_img(img, mask, aug_img, aug_mask)
        # cv2.imwrite("vis.png", vis)

        # Apply Color Augmentation (training mode only)
        if self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                aug_img = Image.fromarray(aug_img)
                aug_img = self.image_aug(aug_img)
                aug_img = np.array(aug_img)

        return aug_img, aug_mask
    
    def __getitem__(self, index):
        load_data = self.total_data[index]
        image_path = load_data["image_path"]
        instance_info = load_data["instance_info"]

        # load image as numpy array RGB
        img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for inst_id, inst in enumerate(instance_info):
            obj_id = inst['obj_id']
            # load mask
            obj_mask = np.array(Image.open(inst['mask_path']))
            mask[obj_mask > 0] = obj_id

        # random crop the image and mask (augmentation)
        if self.random_crop and random.random() < self.random_crop_ratio:
            crop_scale = random.uniform(self.random_crop_scale[0], self.random_crop_scale[1])
            img, mask = random_scaled_crop(img, mask, crop_scale=crop_scale)

        # Keep a copy of the full instance mask before we mask-out unconditioned objects.
        # This will be used to supervise mask proposals for *all* annotated objects.
        mask_all = mask.copy()

        # remove the cropped out objects
        unique_ids = np.unique(np.array(mask))
        # Only keep obj IDs that are present in the mask (exclude background 0)
        unique_obj_ids = set(unique_ids[unique_ids > 0])

        # Collect available per-instance condition images first (skip missing cond dirs)
        cond_imgs = []
        cond_masks = []
        cond_mask_ids = []

        for obj_id in unique_obj_ids:
            cond_img, cond_mask = self.get_cond_data(obj_id)
            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_mask_ids.append(obj_id)
        # load other negative condition images
        for obj_id in self.obj_ids:
            if obj_id in unique_obj_ids:
                continue
            cond_img, cond_mask = self.get_cond_data(obj_id)
            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_mask_ids.append(0)  # negative sample

        cond_imgs = np.stack(cond_imgs, axis=0)  # (N, H, W, 3)
        cond_masks = np.stack(cond_masks)
        cond_imgs = torch.tensor(cond_imgs, dtype=torch.float32)  # (N, H, W, 3)
        cond_masks = torch.tensor(cond_masks, dtype=torch.float32)  # (N, H, W)
        cond_mask_ids = torch.tensor(cond_mask_ids, dtype=torch.long)  # (N,)

        assert cond_imgs.shape[0] == cond_masks.shape[0] == cond_mask_ids.shape[0], (
            f"cond alignment mismatch: cond_imgs={cond_imgs.shape[0]}, cond_masks={cond_masks.shape[0]}, cond_mask_ids={cond_mask_ids.shape[0]}"
        )

        # random shuffle the condition images and mask ids
        perm = torch.randperm(cond_imgs.shape[0])
        cond_imgs = cond_imgs[perm]
        cond_masks = cond_masks[perm]
        cond_mask_ids = cond_mask_ids[perm]

        # double check the mask - mask out the objects not in cond_mask_ids (before shift)
        # Keep only pixels that correspond to actual instances (exclude negative samples with ID=0)
        valid_mask_ids = cond_mask_ids[cond_mask_ids > 0].cpu().numpy()
        valid_mask = np.isin(mask, valid_mask_ids)
        mask = mask * valid_mask

        # shift mask ids to match the augmentation transform (which does -1 shift)
        cond_mask_ids -= 1 # shift with -1

        # transform (image augmentation and mask transform)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        mask_all = Image.fromarray(mask_all)
        img, mask, mask_all = self._img_augmentation(img, mask, mask_all) 

        # check if mask and cond_mask_ids match (only positive samples should be in mask)
        positive_cond_ids = cond_mask_ids[cond_mask_ids >= 0]
        unique_mask = set(torch.unique(mask).tolist())
        # remove background id (-1) from mask before comparing, if present
        if -1 in unique_mask:
            unique_mask.remove(-1)
        assert unique_mask == set(positive_cond_ids.tolist()), f"Mask IDs and cond_mask_ids do not match! mask ids: {torch.unique(mask)}, positive cond_mask_ids: {positive_cond_ids.tolist()}"
        # remap the mask IDs (obj_id-1) to cond_imgs indices (after shuffle)
        semantic_mask = self.remap_mask_ids_to_indices(mask, cond_mask_ids)

        # Sanity: semantic_mask values must be valid indices into cond_imgs or -1.
        if semantic_mask.numel() > 0:
            unique_sem = torch.unique(semantic_mask)
            unique_sem = unique_sem[unique_sem >= 0]
            if unique_sem.numel() > 0:
                assert int(unique_sem.max().item()) < int(cond_mask_ids.numel()), (
                    f"semantic_mask contains index >= num_cond: max={int(unique_sem.max().item())}, num_cond={int(cond_mask_ids.numel())}"
                )
                # Negative templates (cond_mask_ids == -1) must not appear in semantic_mask.
                assert torch.all(cond_mask_ids[unique_sem] >= 0), "semantic_mask points to negative-template indices"
        # generate targets from semantic mask
        targets = self.generate_targets(semantic_mask, bg_val=-1)

        # Targets for mask proposal supervision (all annotated objects)
        # These labels are instance-ids (not condition indices) and are NOT used for class loss.
        targets_all = self.generate_targets(mask_all, bg_val=-1)
        targets["masks_all"] = targets_all["masks"]

        # # Visualzation
        # vis = visualize_targets_and_templates(img, targets, cond_imgs)
        # # vis = visualize_image_mask_and_templates(img, semantic_mask, cond_imgs)
        # os.makedirs("test", exist_ok=True)
        # cv2.imwrite(f'test/vis_mask_{index}.png', vis[:,:,::-1])

        img = torch.from_numpy(np.array(img)).permute(2,0,1).to(torch.uint8) # (3, H, W)
        cond_imgs = cond_imgs.permute(0,3,1,2)  # (N, 3, H, W)
        
        batch_data = {
            "img": img, 
            "targets": targets,
            "semantic_mask": semantic_mask, 
            "cond_imgs": cond_imgs, 
            "cond_masks": cond_masks,
        }
        return batch_data

    def __len__(self):
        return self.len_val

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64') - 1
        return torch.from_numpy(target)

    def remap_mask(self, mask, perm):
        """
        Updates mask indices to match the shuffled cond_imgs order.
        Logic: If cond_imgs was shuffled with `perm`, mask values must map to the new indices.
        """
        # 1. Compute inverse permutation (LUT): old_index -> new_index
        inv_perm = torch.argsort(perm).to(mask.device)

        # 2. Apply mapping
        new_mask = mask.clone()
        valid = mask >= 0  # Ignore background (-1)
        
        # Use mask values as indices into inv_perm to get new values
        new_mask[valid] = inv_perm[mask[valid]]

        return new_mask

    def remap_mask_ids_to_indices(self, mask: torch.Tensor, cond_mask_ids: torch.Tensor) -> torch.Tensor:
        """Map mask values (object IDs) to condition indices.

        Here, `cond_mask_ids` is *not* a permutation. It is an ID list aligned with `cond_imgs`:
        - positive template at index i has ID = obj_id-1 (>= 0)
        - negative templates have ID = -1

        This remaps each pixel label `v` in `mask` (where v is obj_id-1) to the index `i` such
        that `cond_mask_ids[i] == v`. Background (-1) stays -1.
        """
        if cond_mask_ids.numel() == 0:
            return mask.clone()

        new_mask = mask.clone()
        valid = new_mask >= 0
        if not torch.any(valid):
            return new_mask

        max_id = int(torch.max(cond_mask_ids).item())
        if max_id < 0:
            # Only negatives exist.
            new_mask[valid] = -1
            return new_mask

        id_to_index = torch.full((max_id + 1,), -1, dtype=torch.long, device=cond_mask_ids.device)
        pos = cond_mask_ids >= 0
        id_to_index[cond_mask_ids[pos]] = torch.nonzero(pos, as_tuple=False).squeeze(1)

        # Map each pixel ID -> condition index.
        new_mask[valid] = id_to_index[new_mask[valid]]
        return new_mask

    def generate_targets(self, semantic_mask, bg_val=-1):
        """
        Splits a semantic mask into separate binary masks for each class/instance.
        
        Args:
            semantic_mask: (H, W) tensor.
            bg_val: Background value to ignore (default -1).

        Returns:
            targets: Dictionary with keys:
                - "masks": (N, H, W) Tensor (0 or 1).
                - "labels": (N,) Tensor (Class IDs).
        """
        # 1. Get unique class IDs, excluding background
        unique_vals = torch.unique(semantic_mask)
        class_ids = unique_vals[unique_vals != bg_val]

        # Handle case with no foreground objects
        if len(class_ids) == 0:
            return {
                "masks": torch.zeros((0, *semantic_mask.shape), dtype=torch.long, device=semantic_mask.device),
                "labels": class_ids
            }

        # 2. Create binary masks using broadcasting
        # Compare (1, H, W) with (N, 1, 1) -> Result is (N, H, W) Boolean Tensor
        binary_masks = (semantic_mask.unsqueeze(0) == class_ids.view(-1, 1, 1))

        targets = {
            "masks": binary_masks.long(),  # Convert Bool to Long (0/1)
            "labels": class_ids,
        }

        return targets

   
class LMO_Test_Segmentation(data.Dataset):
    def __init__(self, LMODataset):
        self.data_root = LMODataset["data_root"]
        self.render_template_root = LMODataset["render_template_root"]
        self.condition_size = LMODataset["condition_size"]
        self.obj_ids = [1,5,6,8,9,10,11,12]
        
        self.total_data = []
        self.annotation = defaultdict(list)
        scenes = sorted(os.listdir(os.path.join(self.data_root)))
        for scene in scenes:
            scene_rgb_dir = os.path.join(self.data_root, scene, "rgb")
            frames = sorted([f for f in os.listdir(scene_rgb_dir) if f.endswith('.png')])
            scene_gt_path = os.path.join(self.data_root, scene, "scene_gt.json")
            with open(scene_gt_path, 'r') as f: 
                scene_gt = json.load(f)
            for frame in frames:
                frame_id = int(frame.split('.')[0])
                frame_gt = scene_gt[str(frame_id)]
                frame_image_path = os.path.join(scene_rgb_dir, frame)
                instance_info = []
                for inst_id, item in enumerate(frame_gt):
                    obj_id = item['obj_id']
                    obj_mask_path = os.path.join(self.data_root, scene, "mask_visib", f"{frame_id:06d}_{inst_id:06d}.png")
                    instance_info.append({
                        "obj_id": obj_id,
                        "mask_path": obj_mask_path,
                    })
                self.total_data.append({
                    "image_path": frame_image_path,
                    "instance_info": instance_info, 
                    "scene_id": int(scene),
                    "image_id": frame_id,
                    })

        self.len_val = len(self.total_data)                 
        logging.info(f"LMO Data size: {self.len_val}")

    def __getitem__(self, index):
        load_data = self.total_data[index]
        image_path = load_data["image_path"]
        instance_info = load_data["instance_info"]

        # load image as numpy array RGB
        img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB

        # load mask and load condition images for the objects in the image
        cond_imgs = []
        cond_masks = []
        cond_imgs_path = []
        cond_obj_ids = []
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for inst_id, inst in enumerate(instance_info):
            obj_id = inst['obj_id']
            # load mask
            obj_mask = np.array(Image.open(inst['mask_path']))
            mask[obj_mask > 0] = inst_id + 1
            # load condition image
            template_dir = os.path.join(self.render_template_root, f"obj_{obj_id:06d}")
            image_path = glob.glob(os.path.join(template_dir, "*.png"))[0]
            cond_img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
            cond_mask = get_mask(cond_img)
            # resize and crop the condition image
            cond_img, cond_mask = crop_and_resize(cond_img, cond_mask, size=self.condition_size, crop_rel_pad=0)
            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_imgs_path.append(image_path)
            cond_obj_ids.append(obj_id)

        # load other negative condition images
        for obj_id in self.obj_ids:
            if obj_id in [inst['obj_id'] for inst in instance_info]:
                continue
            template_dir = os.path.join(self.render_template_root, f"obj_{obj_id:06d}")
            image_path = glob.glob(os.path.join(template_dir, "*.png"))[0]
            cond_img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
            cond_mask = get_mask(cond_img)
            # resize and crop the condition image
            cond_img, cond_mask = crop_and_resize(cond_img, cond_mask, size=self.condition_size, crop_rel_pad=0)
            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_imgs_path.append(image_path)
            cond_obj_ids.append(obj_id)

        cond_imgs = np.stack(cond_imgs, axis=0)  # (N, H, W, 3)
        cond_masks = np.stack(cond_masks)
        cond_imgs = torch.tensor(cond_imgs, dtype=torch.float32)  # (N, H, W, 3)
        cond_masks = torch.tensor(cond_masks, dtype=torch.float32)  # (N, H, W)

        mask = self._mask_transform(mask)
        
        # random shuffle the condition images
        perm = torch.randperm(cond_imgs.shape[0])
        cond_imgs = cond_imgs[perm]
        cond_masks = cond_masks[perm]
        cond_obj_ids = torch.tensor(cond_obj_ids)[perm]
        # remap the semantic_mask ids to cond_imgs indices
        semantic_mask = self.remap_mask(mask, perm)
        # Conditioned targets (class head supervision)
        targets = self.generate_targets(semantic_mask, bg_val=-1)

        # All-instance targets (mask proposal supervision). Here `mask` already contains
        # the per-instance IDs for objects in the image (no negatives).
        targets_all = self.generate_targets(mask, bg_val=-1)
        targets["masks_all"] = targets_all["masks"]

        # # Visualzation
        # vis = visualize_targets_and_templates(img, targets, cond_imgs)
        # # vis = visualize_image_mask_and_templates(img, semantic_mask, cond_imgs)
        # os.makedirs("test", exist_ok=True)
        # cv2.imwrite(f'test/vis_mask_{index}.png', vis[:,:,::-1])

        img = torch.from_numpy(img.copy()).permute(2,0,1).to(torch.uint8) # (3, H, W)
        cond_imgs = cond_imgs.permute(0,3,1,2)  # (N, 3, H, W)
        
        batch_data = {
            "scene_id": load_data["scene_id"],
            "image_id": load_data["image_id"],
            "img_path": image_path,
            "img": img, 
            "targets": targets, 
            "semantic_mask": semantic_mask,
            "cond_imgs": cond_imgs, 
            "cond_masks": cond_masks,
            "cond_obj_ids": cond_obj_ids
        }
        return batch_data

    def __len__(self):
        return self.len_val

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64') - 1
        return torch.from_numpy(target)

    def remap_mask(self, mask, perm):
        """
        Updates mask indices to match the shuffled cond_imgs order.
        Logic: If cond_imgs was shuffled with `perm`, mask values must map to the new indices.
        """
        # 1. Compute inverse permutation (LUT): old_index -> new_index
        inv_perm = torch.argsort(perm).to(mask.device)

        # 2. Apply mapping
        new_mask = mask.clone()
        valid = mask >= 0  # Ignore background (-1)
        
        # Use mask values as indices into inv_perm to get new values
        new_mask[valid] = inv_perm[mask[valid]]

        return new_mask

    def generate_targets(self, semantic_mask, bg_val=-1):
        """
        Splits a semantic mask into separate binary masks for each class/instance.
        
        Args:
            semantic_mask: (H, W) tensor.
            bg_val: Background value to ignore (default -1).

        Returns:
            targets: Dictionary with keys:
                - "masks": (N, H, W) Tensor (0 or 1).
                - "labels": (N,) Tensor (Class IDs).
        """
        # 1. Get unique class IDs, excluding background
        unique_vals = torch.unique(semantic_mask)
        class_ids = unique_vals[unique_vals != bg_val]

        # Handle case with no foreground objects
        if len(class_ids) == 0:
            return {
                "masks": torch.zeros((0, *semantic_mask.shape), dtype=torch.long, device=semantic_mask.device),
                "labels": class_ids
            }

        # 2. Create binary masks using broadcasting
        # Compare (1, H, W) with (N, 1, 1) -> Result is (N, H, W) Boolean Tensor
        binary_masks = (semantic_mask.unsqueeze(0) == class_ids.view(-1, 1, 1))

        targets = {
            "masks": binary_masks.long(),  # Convert Bool to Long (0/1)
            "labels": class_ids,
        }

        return targets

    
if __name__ == '__main__':
    import yaml
    with open("configs/dinov2/OC/config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    # dataset = LMO_Train_Segmentation(LMODataset=config_dict['LMO_Train_Dataset'], augs=config_dict['augs'])
    dataset = LMO_Test_Segmentation(LMODataset=config_dict['LMO_Test_Dataset'])
    print(len(dataset))
    for i in range(20):
        dataset[i]
