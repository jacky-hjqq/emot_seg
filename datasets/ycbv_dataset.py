import torch.utils.data as data
import os
import json
import logging
import torch
import numpy as np
import cv2
import glob
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from datasets.utils.img_utils import get_mask, crop_and_resize
from datasets.utils.vis_utils import *

class YCBVSegmentation(data.Dataset):
    def __init__(self, YCBVDataset):
        self.data_root = YCBVDataset["data_root"]
        self.render_template_root = YCBVDataset["render_template_root"]
        self.condition_size = YCBVDataset["condition_size"]

        self.obj_ids = list(range(1, 22))
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
        logging.info(f"YCBV Data size: {self.len_val}")

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
            cond_img, cond_mask = crop_and_resize(cond_img, cond_mask, size=self.condition_size, crop_rel_pad=0.2)
            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_imgs_path.append(image_path)

        # load other negative condition images
        for obj_id in self.obj_ids:
            if obj_id in [inst['obj_id'] for inst in instance_info]:
                continue
            template_dir = os.path.join(self.render_template_root, f"obj_{obj_id:06d}")
            image_path = glob.glob(os.path.join(template_dir, "*.png"))[0]
            cond_img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
            cond_mask = get_mask(cond_img)
            # resize and crop the condition image
            cond_img, cond_mask = crop_and_resize(cond_img, cond_mask, size=self.condition_size, crop_rel_pad=0.2)
            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_imgs_path.append(image_path)

        cond_imgs = np.stack(cond_imgs, axis=0)  # (N, H, W, 3)
        cond_masks = np.stack(cond_masks)
        cond_imgs = torch.tensor(cond_imgs, dtype=torch.float32)  # (N, H, W, 3)
        cond_masks = torch.tensor(cond_masks, dtype=torch.float32)  # (N, H, W)

        mask = self._mask_transform(mask)
        
        # random shuffle the condition images
        perm = torch.randperm(cond_imgs.shape[0])
        cond_imgs = cond_imgs[perm]
        cond_masks = cond_masks[perm]
        # remap the semantic_mask ids to cond_imgs indices
        semantic_mask = self.remap_mask(mask, perm)
        # generate targets from semantic mask
        targets = self.generate_targets(semantic_mask, bg_val=-1)

        # # Visualzation
        # vis = visualize_targets_and_templates(img, targets, cond_imgs)
        # cv2.imwrite('vis_targets.png', vis[:,:,::-1])
        # vis = visualize_image_mask_and_templates(img, semantic_mask, cond_imgs)
        # cv2.imwrite('vis_mask.png', vis[:,:,::-1])

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