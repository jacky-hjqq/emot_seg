import torch.utils.data as data
import random
import os
import json
import logging
import torch
import numpy as np
import cv2
from PIL import Image
from imageio import imread
from collections import defaultdict
from torchvision import transforms
from datasets.utils.img_utils import get_mask, crop_and_resize
from datasets.utils.augmentation import (
    get_image_augmentation,
    random_rotation,
    random_scaled_crop
)
from datasets.utils.vis_utils import *

class Omni6DSegmentation(data.Dataset):
    def __init__(self, Omni6dDataset, augs=None, scene_name=None):
        self.data_root = Omni6dDataset['data_root']
        self.model_meta_root = Omni6dDataset['model_meta_root']
        self.condition_size = Omni6dDataset.get('condition_size', 224)

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
            
        if scene_name is None:
            self.scene_name = ['ikea', 'scannet++'] ###########
        else:
            self.scene_name = [scene_name]
            
        model_meta_path = os.path.join(self.model_meta_root, 'obj_meta.json')
        with open(model_meta_path, 'r') as f:
            self.model_meta = json.load(f)

        self.color_tpath = "{data_root}/{scene_patch}/{scene_name}/{scene_id:04d}/{frame_id:04d}_color.png"
        self.mask_tpath = "{data_root}/{scene_patch}/{scene_name}/{scene_id:04d}/{frame_id:04d}_mask.exr"
        self.meta_tpath = "{data_root}/{scene_patch}/{scene_name}/{scene_id:04d}/{frame_id:04d}_meta.json"

        self.total_data = []
        self.total_inst = []
        self.annotation_labels = defaultdict(list)
        self.annotation_objects = defaultdict(list)
        self.obj_label_ids = []
        data_patches = sorted(os.listdir(os.path.join(self.data_root)))
        for data_patch in data_patches:
            data_patch_dir = os.path.join(self.data_root, data_patch)
            for scene_name in self.scene_name:
                scene_ids = sorted(os.listdir(os.path.join(data_patch_dir, scene_name)))
                for scene_id in scene_ids:
                    scene_dir = os.path.join(data_patch_dir, scene_name, scene_id)
                    frames = sorted([f for f in os.listdir(scene_dir) if f.endswith('_color.png')])
                    for frame in frames:
                        frame_id = frame.split('_')[0]  # Assuming the frame name is like '0001_color.png'
                        paths = self.get_frame_paths(data_patch, scene_name, scene_id, frame_id)
                        if paths is None:
                            continue # skip frames with missing files
                        with open(paths["meta"], 'r') as f:
                            meta_data = json.load(f)
                        instance_info = []
                        for obj_key in meta_data['objects']:
                            obj_meta_data = meta_data['objects'][obj_key]
                            obj_id = obj_meta_data['meta']['oid']
                            inst_id = int(obj_key.split('_', 1)[0])
                            obj_label = obj_meta_data['meta']['class_name']
                            obj_label_id = obj_meta_data['meta']['class_label']

                            if obj_label_id not in self.obj_label_ids:
                                self.obj_label_ids.append(obj_label_id)

                            instance_info.append({
                                "inst_id": inst_id,
                                "obj_id": obj_id,
                                "obj_label": obj_label,
                                "obj_label_id": obj_label_id,
                            })

                            data_idx = len(self.total_inst)
                            self.total_inst.append({
                                "data_idx": data_idx,
                                "image_path": paths["color"],
                                "mask_path": paths["mask"],
                                "inst_id": inst_id,
                                "obj_name": obj_id,
                                "obj_label": obj_label,
                                "obj_label_id": obj_label_id,
                                })
                            self.annotation_labels[obj_label_id].append({
                                "data_idx": data_idx,
                            })
                            self.annotation_objects[obj_id].append({
                                "data_idx": data_idx,
                            })

                        self.total_data.append({
                            "image_path": paths["color"],
                            "mask_path": paths["mask"],
                            "instance_info": instance_info, 
                            })

        self.len_train = len(self.total_data)                 
        logging.info(f"Omni6D Data size: {self.len_train}")
    
    def __len__(self):
        return self.len_train

    def __getitem__(self, index):
        load_data = self.total_data[index]
        image_path = load_data["image_path"]
        mask_path = load_data["mask_path"]
        instance_info = load_data["instance_info"]

        # load image and mask
        img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
        mask = (imread(mask_path)[:, :, 0] * 255).astype(np.uint8)

        # # random crop the image and mask (augmentation)
        # if self.random_crop and random.random() < self.random_crop_ratio:
        #     crop_scale = random.uniform(self.random_crop_scale[0], self.random_crop_scale[1])
        #     img, mask = random_scaled_crop(img, mask, crop_scale=crop_scale)
            # # remove the small objects in the mask after crop
            # unique_ids = np.unique(mask)
            # for uid in unique_ids:
            #     if uid == 0:
            #         continue
            #     obj_mask = (mask == uid).astype(np.uint8)
            #     # remove the object if less than 1% of the image area
            #     if np.sum(obj_mask) < (0.001 / crop_scale) * (img.shape[0] * img.shape[1]):
            #         mask[mask == uid] = 0

            # vis = visualize_cond_img(img, mask, img, mask)
            # os.makedirs("test", exist_ok=True)
            # cv2.imwrite(f"test/vis_{index}.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # remove the cropped out objects
        unique_ids = np.unique(np.array(mask))
        # Only keep instance IDs that are present in the mask (exclude background 0)
        unique_inst_ids = set(unique_ids[unique_ids > 0])

        # load the condition images based on the labels
        labels_id_im = [inst['obj_label_id'] for inst in instance_info if inst['inst_id'] in unique_inst_ids]
        labels_id_non_im = list(set(self.obj_label_ids) - set(labels_id_im))
        # random sample non-im objects
        num_condition_imgs = 30
        num_non_im = max(0, num_condition_imgs - len(labels_id_im))
        num_non_im = min(num_non_im, len(labels_id_non_im))  # Can't sample more than available
        labels_id_non_im = random.sample(labels_id_non_im, num_non_im) if num_non_im > 0 else []

        cond_imgs = []
        cond_masks = []
        cond_mask_ids = []
        # load condition images for the objects in the image
        for inst in instance_info:
            # Only load condition images for instances that exist in the current mask
            if inst['inst_id'] not in unique_inst_ids:
                continue
            mask_id = inst['inst_id']
            obj_id = inst['obj_id']
            # random sample one from the self.annotation[obj_label_id]
            while True:
                cond_idx = random.choice(self.annotation_objects[obj_id])
                cond_data = self.total_inst[cond_idx['data_idx']]
                cond_img, cond_mask = self.get_cond_data(cond_data)
                if cond_img is not None:
                    break
            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_mask_ids.append(mask_id)
        # load condition images for the non-im objects
        for obj_label_id in labels_id_non_im:
            # random sample one from the self.annotation[obj_label_id]
            while True:
                cond_idx = random.choice(self.annotation_labels[obj_label_id])
                cond_data = self.total_inst[cond_idx['data_idx']]
                cond_img, cond_mask = self.get_cond_data(cond_data)
                if cond_img is not None:
                    break

            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_mask_ids.append(0)  # non-im object mask id = 0

        cond_imgs = np.stack(cond_imgs, axis=0)  # (N, H, W, 3)
        cond_masks = np.stack(cond_masks)
        cond_imgs = torch.tensor(cond_imgs, dtype=torch.float32)  # (N, H, W, 3)
        cond_masks = torch.tensor(cond_masks, dtype=torch.float32)  # (N, H, W)
        cond_mask_ids = torch.tensor(cond_mask_ids, dtype=torch.long)  # (N,)

        # random shuffle the condition images and mask ids
        perm = torch.randperm(cond_imgs.shape[0])
        cond_imgs = cond_imgs[perm]
        cond_masks = cond_masks[perm]
        cond_mask_ids = cond_mask_ids[perm]

        # double check the mask - mask out the objects not in cond_mask_ids (before shift)
        # Keep only pixels that correspond to actual instances (exclude negative samples with ID=0)
        valid_mask_ids = cond_mask_ids[cond_mask_ids > 0].numpy()
        valid_mask = np.isin(mask, valid_mask_ids)
        mask = mask * valid_mask

        # shift mask ids to match the augmentation transform (which does -1 shift)
        cond_mask_ids -= 1 # shift with -1

        # transform (image augmentation and mask transform)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        img, mask = self._img_augmentation(img, mask)   
        # check if mask and cond_mask_ids match (only positive samples should be in mask)
        positive_cond_ids = cond_mask_ids[cond_mask_ids >= 0]
        assert set(torch.unique(mask).tolist()) == set(positive_cond_ids.tolist() + [-1]), f"Mask IDs and cond_mask_ids do not match! mask ids: {torch.unique(mask)}, positive cond_mask_ids: {positive_cond_ids.tolist()}"
        # remap the mask ids to cond_imgs indices
        semantic_mask = self.remap_mask(mask, cond_mask_ids)
        # generate targets from semantic mask
        targets = self.generate_targets(semantic_mask, bg_val=-1)

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
    
    def get_frame_paths(self, scene_patch: str, scene_name: str, scene_id: int, frame_id: int) -> dict:
        """
        Returns a dict with file paths for color, depth, mask, and meta
        for the given scene_patch, scene_id, and frame_id.
        """
        params = {
            "data_root":   self.data_root,
            "scene_patch": scene_patch,
            "scene_name":  scene_name,
            "scene_id":    int(scene_id),
            "frame_id":    int(frame_id),
        }

        paths = {
            "color": self.color_tpath.format(**params),
            "mask":  self.mask_tpath.format(**params),
            "meta":  self.meta_tpath.format(**params),
        }

        # check that every file exists
        if not all(os.path.exists(p) for p in paths.values()):
            return None

        return paths
    
    def get_cond_data(self, cond_data):
        image_path = cond_data['image_path']
        cond_img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
        mask_path = cond_data['mask_path']
        cond_mask = (imread(mask_path)[:, :, 0] * 255).astype(np.uint8)
        is_target = (cond_mask == cond_data['inst_id'])
        if is_target.sum() < 100:
            return None, None
            
        cond_mask = is_target.astype(np.uint8) * 255
        # # cond_img augmentation
        # cond_img, cond_mask = self._cond_img_augmentation(cond_img, cond_mask)
        # resize and crop
        cond_img, cond_mask = crop_and_resize(cond_img, cond_mask, size=self.condition_size, crop_rel_pad=0.2)
        # mask the cond_img with the cond_mask (Black)
        cond_img = cond_img * cond_mask[:, :, None].astype(bool)

        return cond_img, cond_mask
    
    def _img_augmentation(self, img, mask):
        # Random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # Apply Color Augmentation (training mode only)
        if self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                img = self.image_aug(img)

        return img, self._mask_transform(mask)
    
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

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64') - 1
        return torch.from_numpy(target)
    
    def remap_mask(self, mask, cond_mask_ids):
        """
        Remaps values in the mask to their corresponding index in cond_mask_ids.
        
        Args:
            mask (torch.Tensor): The original segmentation mask (H, W).
            cond_mask_ids (torch.Tensor): 1D tensor of class IDs. 
                                        cond_mask_ids[i] is the class ID that should map to index i.
        
        Returns:
            torch.Tensor: The remapped mask where values are 0, 1, 2... corresponding to the index in cond_mask_ids.
                        Pixels not found in cond_mask_ids (or originally -1) are set to -1.
        """
        
        # 1. Find the maximum ID value to determine the necessary size of the Lookup Table (LUT).
        # The LUT must be large enough to use the largest original ID as an array index.
        max_id = cond_mask_ids.max()

        # 2. Initialize the Lookup Table with -1 (background/ignore value).
        # We use (max_id + 1) because indices are 0-based.
        lut = torch.full((max_id + 1,), -1, dtype=torch.long, device=mask.device)

        # 3. Populate the Lookup Table.
        # Logic: lut[original_id] = new_index
        # We map the values in cond_mask_ids to a sequence [0, 1, 2, ... len-1].
        lut[cond_mask_ids] = torch.arange(len(cond_mask_ids), device=mask.device)

        # 4. Apply the Lookup Table to the mask.
        new_mask = mask.clone()

        # Identify valid pixels (exclude -1 from the original mask).
        # We must filter these out because -1 cannot be used as an index to query the 'lut'.
        valid_mask = mask >= 0
        
        # Perform the mapping: use the pixel value as the index into the LUT.
        # If mask[x, y] is 24, we fetch lut[24] (which is 0) and assign it to new_mask.
        new_mask[valid_mask] = lut[mask[valid_mask]]

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
    
