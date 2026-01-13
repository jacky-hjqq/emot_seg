import torch.utils.data as data
import random
import os
import json
import logging
import torch
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
from torchvision import transforms
import sys
sys.path.append('.')
from datasets.utils.img_utils import get_mask, crop_and_resize
from datasets.utils.augmentation import (
    get_image_augmentation,
    random_rotation,
    random_scaled_crop
)
from datasets.utils.vis_utils import *

class GSOSegmentation(data.Dataset):
    def __init__(self, GSODataset, augs=None):
        self.data_root = GSODataset["data_root"]
        self.cond_root = GSODataset["cond_root"]
        self.condition_size = int(GSODataset["condition_size"]) 
        self.num_condition_imgs = 30
        
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

        color_tpath = "{data_root}/{data_patch}/{scene_id}/{scene_name}/{render_name}/rgb/rgb_000000.png"
        mask_tpath = "{data_root}/{data_patch}/{scene_id}/{scene_name}/{render_name}/instance_segmentation/instance_segmentation_000000.png"
        instance_map_tpath = "{data_root}/{data_patch}/{scene_id}/{scene_name}/{render_name}/instance_segmentation/instance_segmentation_mapping_000000.json"
        instance_semantic_map_tpath = "{data_root}/{data_patch}/{scene_id}/{scene_name}/{render_name}/instance_segmentation/instance_segmentation_semantics_mapping_000000.json"
        
        self.total_data = []
        self.total_inst = []
        self.obj_classes = []
        self.annotation_classes = defaultdict(list)
        self.annotation_objects= defaultdict(list)
        data_patches = sorted(os.listdir(os.path.join(self.data_root)))
        for data_patch in data_patches:
            scene_ids = sorted(os.listdir(os.path.join(self.data_root, data_patch)))
            for scene_id in scene_ids:
                scene_names = sorted(os.listdir(os.path.join(self.data_root, data_patch, scene_id)))
                for scene_name in scene_names:
                    render_names = sorted(os.listdir(os.path.join(self.data_root, data_patch, scene_id, scene_name)))
                    for render_name in render_names:
                        paths = self.get_paths(
                            data_root=self.data_root,
                            data_patch=data_patch,
                            scene_id=scene_id,
                            scene_name=scene_name,
                            render_name=render_name,
                            color_tpath=color_tpath,
                            mask_tpath=mask_tpath,
                            instance_map_tpath=instance_map_tpath,
                            instance_semantic_map_tpath=instance_semantic_map_tpath,
                        )
                        if paths is None:
                            continue

                        # load rgb and mask
                        # # visualize instance
                        # mask = cv2.imread(paths["mask"], cv2.IMREAD_UNCHANGED).astype(np.uint8)
                        # mask_ids = np.unique(mask)
                        # rgb = cv2.imread(paths["color"])[:,:,::-1]  # BGR to RGB
                        # visualize_instance(rgb, mask, inst_id=4, save_path="instance_viz.png", alpha=0.5)

                        # load instance map 
                        with open(paths["instance_map"], "r") as f:
                            instance_map = json.load(f)
                        # load instance semantic map
                        with open(paths["instance_semantic_map"], "r") as f:
                            instance_semantic_map = json.load(f)
                        instance_info = []
                        for key in instance_map:
                            inst_value = instance_map[key]
                            inst_value = inst_value.lower()
                            # skip background and unlabeled
                            if inst_value in ["background", "unlabelled"]:
                                continue
                            # skip if inst not object
                            value_type = inst_value.split("/")[2]
                            if value_type != "objects":
                                continue
                            obj_name = inst_value.split("/")[-3]
                            obj_class = instance_semantic_map[key]['class'].lower()
                            if obj_class not in self.obj_classes:
                                self.obj_classes.append(obj_class)
                            data_idx = len(self.total_inst)
                            instance_info.append({
                                "obj_class": obj_class,
                                "obj_name": obj_name,
                                "inst_id": int(key),
                            })
                            self.total_inst.append(
                                {
                                    "data_idx": data_idx,
                                    "image_path": paths["color"],
                                    "mask_path": paths["mask"],
                                    "inst_id": int(key),
                                    "obj_class": obj_class,
                                    "obj_name": obj_name,
                                }
                            )
                            self.annotation_classes[obj_class].append({"data_idx": data_idx})
                            self.annotation_objects[obj_name].append({"data_idx": data_idx})

                        self.total_data.append({
                            "image_path": paths['color'],
                            "mask_path": paths['mask'],
                            "instance_info": instance_info,
                        })

            # break

        self.len_train = len(self.total_data)                 
        logging.info(f"GSO Data size: {self.len_train}")
    
    def _collect_pos_conditions(self, instance_info, unique_inst_ids):
        """Collect available per-instance condition images.

        Returns a list of dicts with keys:
            - 'inst_id', 'obj_id', 'obj_label_id', 'cond_img', 'cond_mask'
        Only includes instances whose `cond_root/<obj_id>` directory exists and
        where `get_cond_data` returned a valid pair.
        """
        collected = []
        shuffled_instance_info = list(instance_info)
        random.shuffle(shuffled_instance_info)
        for inst in shuffled_instance_info:
            if inst['inst_id'] not in unique_inst_ids:
                continue

            obj_name = inst['obj_name']
            obj_dir = os.path.join(self.cond_root, str(obj_name))
            if not os.path.exists(obj_dir):
                continue
            cond_img, cond_mask = self.get_cond_data(obj_name)
            if cond_img is None:
                continue
            collected.append({
                'inst_id': inst['inst_id'],
                'obj_class': inst['obj_class'],
                'obj_name': obj_name,
                'cond_img': cond_img,
                'cond_mask': cond_mask,
            })
        return collected
        
    def __len__(self):
        return self.len_train

    def __getitem__(self, index):
        load_data = self.total_data[index]
        image_path = load_data["image_path"]
        mask_path = load_data["mask_path"]
        instance_info = load_data["instance_info"]

        # load image and mask
        img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)

        # random crop the image and mask (augmentation)
        if self.random_crop and random.random() < self.random_crop_ratio:
            crop_scale = random.uniform(self.random_crop_scale[0], self.random_crop_scale[1])
            img, mask = random_scaled_crop(img, mask, crop_scale=crop_scale)
        
        # Keep a copy of the full instance mask before we mask-out unconditioned objects.
        # This will be used to supervise mask proposals for *all* annotated objects.
        mask_all = mask.copy()

        # remove the cropped out objects
        unique_ids = np.unique(np.array(mask))
        # Only keep instance IDs that are present in the mask (exclude background 0)
        unique_inst_ids = set(unique_ids[unique_ids > 0])

        # Collect available per-instance condition images first (skip missing cond dirs)
        cond_imgs = []
        cond_masks = []
        cond_mask_ids = []

        primary = self._collect_pos_conditions(instance_info, unique_inst_ids)
        # add primary collected conditions
        for p in primary:
            cond_imgs.append(p['cond_img'])
            cond_masks.append(p['cond_mask'])
            cond_mask_ids.append(p['inst_id'])

        # determine how many non-im (negative) conditions are still needed
        num_primary = len(cond_imgs)
        num_non_im_needed = max(0, self.num_condition_imgs - num_primary)

        # choose candidate classes excluding those already represented by primary collected
        classes_im = set(p['obj_class'] for p in primary)
        candidate_classes = list(set(self.obj_classes) - classes_im)
        random.shuffle(candidate_classes)
        classes_non_im = candidate_classes[:num_non_im_needed] if num_non_im_needed > 0 else []

        # sample one cond per selected label (best-effort with retries)
        for obj_class in classes_non_im:
            while True:
                cond_idx = random.choice(self.annotation_classes[obj_class])
                cond_data = self.total_inst[cond_idx['data_idx']]
                obj_id = cond_data['obj_name']
                obj_dir = os.path.join(self.cond_root, str(obj_id))
                if not os.path.exists(obj_dir):
                    continue
                cond_img, cond_mask = self.get_cond_data(obj_id)
                break

            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_mask_ids.append(0)  # non-im object mask id = 0
        
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
        valid_mask_ids = cond_mask_ids[cond_mask_ids > 0].numpy()
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
        # remap the mask ids to cond_imgs indices
        semantic_mask = self.remap_mask(mask, cond_mask_ids)

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

    def get_paths(
        self,
        data_root: str,
        data_patch: str,
        scene_id: str,
        scene_name: str,
        render_name: str,
        color_tpath: str,
        mask_tpath: str,
        instance_map_tpath: str,
        instance_semantic_map_tpath: str,   
    ) -> dict | None:
        params = {
            "data_root": data_root,
            "data_patch": data_patch,
            "scene_id": scene_id,
            "scene_name": scene_name,
            "render_name": render_name,
        }
        paths = {
            "color": color_tpath.format(**params),
            "mask": mask_tpath.format(**params),
            "instance_map": instance_map_tpath.format(**params),
            "instance_semantic_map": instance_semantic_map_tpath.format(**params),
        }
        if not all(os.path.exists(p) for p in paths.values()):
            return None
        return paths

    
    def get_cond_data(self, obj_name):
        """Load a condition image/mask pair for the given ``obj_name``.
        - Prefers: rgb_<id>.png and mask_<id>.png
        """
        obj_dir = os.path.join(self.cond_root, str(obj_name))
        
        # Check if directory exists to avoid os.listdir errors
        if not os.path.isdir(obj_dir):
            print(f"Directory {obj_dir} does not exist.")
            return None, None

        # Find all files starting with 'rgb_'
        img_files = [f for f in os.listdir(obj_dir) if f.startswith('rgb_') and f.endswith('.png')]

        if len(img_files) > 0:
            img_file = random.choice(img_files)
            
            # Extract the unique ID/suffix by removing 'rgb_' 
            # Example: 'rgb_001.png' -> '001.png'
            suffix = img_file[len('rgb_'):]
            mask_file = 'mask_' + suffix
            
            img_path = os.path.join(obj_dir, img_file)
            mask_path = os.path.join(obj_dir, mask_file)

            if os.path.exists(img_path) and os.path.exists(mask_path):
                cond_img = cv2.imread(img_path)[:, :, ::-1] # BGR to RGB
                cond_mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

                if cond_mask_raw is not None:
                    # Support single-channel or multi-channel masks
                    if cond_mask_raw.ndim == 3:
                        cond_mask = cond_mask_raw[:, :, 0]
                    else:
                        cond_mask = cond_mask_raw
                    
                    # Augmentation
                    cond_img, cond_mask = self._cond_img_augmentation(cond_img, cond_mask)
                    
                    # Binarize and mask the image
                    cond_mask = (cond_mask != 0).astype(np.uint8) * 255
                    cond_img = cond_img * cond_mask[:, :, None].astype(bool)
                    
                    return cond_img, cond_mask
        
        print(f"No valid precomputed pairs for {obj_name} (Expected rgb_*.png and mask_*.png)")
        return None, None
    
    # def get_cond_data(self, obj_name):
    #     """Load a condition image/mask pair for the given ``obj_name``.

    #     - Prefer precomputed condition images under ``self.cond_root/<obj_name>/`` with files
    #       named ``<stem>_image.png`` and ``<stem>_mask.png`` (randomly choose one).
    #     - If no precomputed files are available, randomly sample one annotation instance from
    #       ``self.annotation_objects[obj_id]`` and extract the instance mask from that frame.
    #     Returns ``(cond_img, cond_mask)`` or ``(None, None)`` if no valid source exists.
    #     """
    #     obj_dir = os.path.join(self.cond_root, str(obj_name))
    #     # find all candidate image files that follow the naming convention
    #     img_files = [f for f in os.listdir(obj_dir) if f.endswith('_image.png')]
    #     if len(img_files) > 0:
    #         img_file = random.choice(img_files)
    #         stem = img_file[:-len('_image.png')]
    #         img_path = os.path.join(obj_dir, img_file)
    #         mask_path = os.path.join(obj_dir, stem + '_mask.png')
    #         if os.path.exists(img_path) and os.path.exists(mask_path):
    #             cond_img = cv2.imread(img_path)[:, :, ::-1]
    #             cond_mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    #             # support single-channel or multi-channel masks
    #             if cond_mask_raw is not None:
    #                 if cond_mask_raw.ndim == 3:
    #                     cond_mask = cond_mask_raw[:, :, 0]
    #                 else:
    #                     cond_mask = cond_mask_raw
    #                 # augmentation
    #                 cond_img, cond_mask = self._cond_img_augmentation(cond_img, cond_mask)
    #                 # binarize: consider non-zero as foreground and accept as-is
    #                 cond_mask = (cond_mask != 0).astype(np.uint8) * 255
    #                 cond_img = cond_img * cond_mask[:, :, None].astype(bool)
    #                 return cond_img, cond_mask
    #     else:
    #         print(f"No precomputed cond images for obj {obj_name}, sampling from dataset.")
    #         return None, None
    
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
        
        # Important: cond_mask_ids may contain -1 (negative templates / background after shift).
        # Using -1 to index a LUT will silently index the last element and corrupt the mapping.
        device = mask.device
        cond_mask_ids = cond_mask_ids.to(device)

        pos = cond_mask_ids >= 0
        if not torch.any(pos):
            return torch.full_like(mask, -1)

        pos_ids = cond_mask_ids[pos]
        max_id = int(pos_ids.max().item())

        # LUT maps original mask-id -> index into cond_imgs.
        lut = torch.full((max_id + 1,), -1, dtype=torch.long, device=device)
        all_indices = torch.arange(len(cond_mask_ids), device=device)
        lut[pos_ids] = all_indices[pos]

        new_mask = mask.clone()

        # Only ids within LUT range are indexable; everything else becomes -1.
        valid = (mask >= 0) & (mask <= max_id)
        new_mask[valid] = lut[mask[valid]]
        new_mask[(mask < 0) | (mask > max_id)] = -1

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
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    with open("configs/dinov2/OC/config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    dataset = GSOSegmentation(GSODataset=config_dict['GSODataset'], augs=config_dict['augs'])
    print(len(dataset))
    for i in range(20):
        dataset[i]