import random
import os
import json
import logging
import argparse
import torch
import numpy as np
import cv2
from collections import defaultdict
from imageio import imread
import yaml
from tqdm import tqdm
import shutil
from datasets.utils.img_utils import crop_and_resize
from datasets.utils.vis_utils import *

def get_cond_data(cond_data, condition_size=224):
    image_path = cond_data['image_path']
    cond_img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
    mask_path = cond_data['mask_path']
    cond_mask = (cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:, :, 2] * 255).astype(np.uint8)
    is_target = (cond_mask == cond_data['inst_id'])
    if is_target.sum() < 100:
        return None, None
        
    cond_mask = is_target.astype(np.uint8) * 255
    # # cond_img augmentation
    # cond_img, cond_mask = self._cond_img_augmentation(cond_img, cond_mask)
    # resize and crop
    cond_img, cond_mask = crop_and_resize(cond_img, cond_mask, size=condition_size, crop_rel_pad=0.2)
    # mask the cond_img with the cond_mask (Black)
    cond_img = cond_img * cond_mask[:, :, None].astype(bool)

    return cond_img, cond_mask

def get_frame_paths(
    *,
    data_root: str,
    color_tpath: str,
    mask_tpath: str,
    meta_tpath: str,
    scene_patch: str,
    split: str,
    scene_name: str,
    scene_id: int,
    frame_id: int,
) -> dict | None:
    params = {
        "data_root": data_root,
        "scene_patch": scene_patch,
        "split": split,
        "scene_name": scene_name,
        "scene_id": int(scene_id),
        "frame_id": int(frame_id),
    }
    paths = {
        "color": color_tpath.format(**params),
        "mask": mask_tpath.format(**params),
        "meta": meta_tpath.format(**params),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return paths


def build_omni6d_index(
    *,
    data_root: str,
    model_meta_root: str,
    scene_names: list[str],
):
    model_meta_path = os.path.join(model_meta_root, "obj_meta.json")
    with open(model_meta_path, "r") as f:
        model_meta = json.load(f)

    color_tpath = "{data_root}/{scene_patch}/{split}/{scene_name}/{scene_id:04d}/{frame_id:04d}_color.png"
    mask_tpath = "{data_root}/{scene_patch}/{split}/{scene_name}/{scene_id:04d}/{frame_id:04d}_mask.exr"
    meta_tpath = "{data_root}/{scene_patch}/{split}/{scene_name}/{scene_id:04d}/{frame_id:04d}_meta.json"

    total_inst: list[dict] = []
    annotation_objects: dict[str, list[dict]] = defaultdict(list)

    splits = ['train', 'test']

    data_patches = sorted(os.listdir(os.path.join(data_root)))
    for split in splits:
        for data_patch in data_patches:
            data_patch_dir = os.path.join(data_root, data_patch)
            for scene_name in scene_names:
                scene_root = os.path.join(data_patch_dir, split, scene_name)
                if not os.path.exists(scene_root):
                    continue
                scene_ids = sorted(os.listdir(scene_root))
                for scene_id in scene_ids:
                    scene_dir = os.path.join(scene_root, scene_id)
                    frames = sorted([f for f in os.listdir(scene_dir) if f.endswith("_color.png")])
                    for frame in frames:
                        frame_id = frame.split("_")[0]
                        paths = get_frame_paths(
                            data_root=data_root,
                            color_tpath=color_tpath,
                            mask_tpath=mask_tpath,
                            meta_tpath=meta_tpath,
                            scene_patch=data_patch,
                            split=split,
                            scene_name=scene_name,
                            scene_id=int(scene_id),
                            frame_id=int(frame_id),
                        )
                        if paths is None:
                            continue

                        with open(paths["meta"], "r") as f:
                            meta_data = json.load(f)

                        for obj_key in meta_data["objects"]:
                            obj_meta_data = meta_data["objects"][obj_key]
                            # skip if it's transparent or specular 
                            if 'transparent' in obj_meta_data["material"] or 'specular' in obj_meta_data["material"]:
                                continue
                            obj_id = obj_meta_data["meta"]["oid"]
                            inst_id = int(obj_key.split("_", 1)[0])
                            obj_label = obj_meta_data["meta"]["class_name"]
                            obj_label_id = int(obj_meta_data["meta"]["class_label"])
                            data_idx = len(total_inst)
                            total_inst.append(
                                {
                                    "data_idx": data_idx,
                                    "image_path": paths["color"],
                                    "mask_path": paths["mask"],
                                    "inst_id": inst_id,
                                    "obj_name": obj_id,
                                    "obj_label": obj_label,
                                    "obj_label_id": obj_label_id,
                                    "data_patch": data_patch,
                                    "scene_id": int(scene_id),
                                    "frame_id": int(frame_id),
                                    "scene_name": scene_name,
                                }
                            )
                            annotation_objects[obj_id].append({"data_idx": data_idx})

    return {
        "model_meta": model_meta,
        "total_inst": total_inst,
        "annotation_objects": annotation_objects,
    }


def _sanitize_dir_name(name: str) -> str:
    name = str(name)
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    name = name.strip().strip(".")
    return name or "unknown"


def _load_instance_mask(mask_path: str, inst_id: int) -> np.ndarray:
    # Omni6D masks are stored in EXR; historically we used channel 0 * 255 -> uint8 ids.
    m = imread(mask_path)
    if m.ndim == 3:
        m = m[:, :, 0]
    mask_u8 = (m * 255).astype(np.uint8)
    return mask_u8 == np.uint8(inst_id)


def save_topn_per_object(
    *,
    index: dict,
    out_dir: str,
    top_n: int,
    min_area: int,
    condition_size: int,
    crop_rel_pad: float,
):
    os.makedirs(out_dir, exist_ok=True)

    total_inst = index["total_inst"]
    annotation_objects = index["annotation_objects"]

    obj_iter = annotation_objects.items()
    if tqdm is not None:
        obj_iter = tqdm(list(obj_iter), desc="Objects", unit="obj")

    for obj_id, entries in obj_iter:
        # check if already have top_n samples
        obj_dir = os.path.join(out_dir, _sanitize_dir_name(obj_id))
        if os.path.exists(obj_dir):
            existing_files = os.listdir(obj_dir)
            existing_count = len([f for f in existing_files if f.endswith("_image.png")])
            if existing_count >= int(top_n):
                # logging.info("Skip %s: already have %d samples", obj_id, existing_count)
                continue
        else:
            # delete existing folder and create new
            shutil.rmtree(obj_dir) if os.path.exists(obj_dir) else None
            os.makedirs(obj_dir, exist_ok=True)

        # 1) Read all masks once to compute areas
        scored: list[tuple[int, int]] = []  # (area, data_idx)
        for entry in entries:
            data_idx = int(entry["data_idx"])
            data = total_inst[data_idx]
            inst_id = int(data["inst_id"])
            try:
                mask_bool = _load_instance_mask(data["mask_path"], inst_id)
            except Exception as e:
                logging.warning("Skip %s idx=%d: failed reading mask (%s)", obj_id, data_idx, e)
                continue

            area = int(mask_bool.sum())
            if area < int(min_area):
                continue
            scored.append((area, data_idx))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        selected_data_idx = [data_idx for _area, data_idx in scored[: int(top_n)]]

        # 2) Save selected image+mask into per-object folder
        for rank, data_idx in enumerate(selected_data_idx):
            data = total_inst[data_idx]
            inst_id = int(data["inst_id"])
            image_path = data["image_path"]
            mask_path = data["mask_path"]

            img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logging.warning("Skip %s idx=%d: failed reading image", obj_id, data_idx)
                continue

            img_rgb = img_bgr[:, :, ::-1]

            try:
                mask_bool = _load_instance_mask(mask_path, inst_id)
            except Exception as e:
                logging.warning("Skip %s idx=%d: failed re-reading mask (%s)", obj_id, data_idx, e)
                continue

            mask_u8 = (mask_bool.astype(np.uint8) * 255)

            # Resize + crop to match training templates, then apply the mask.
            cond_img, cond_mask = crop_and_resize(
                img_rgb,
                mask_u8,
                size=int(condition_size),
                crop_rel_pad=float(crop_rel_pad),
            )
            cond_mask = cond_mask.astype(np.uint8)
            cond_img = cond_img * cond_mask[:, :, None].astype(bool)

            stem = f"{rank:03d}_{data['data_patch']}_{data['scene_id']}_{data['scene_name']}_{data['frame_id']}"
            img_out = os.path.join(obj_dir, stem + "_image.png")
            mask_out = os.path.join(obj_dir, stem + "_mask.png")

            cv2.imwrite(img_out, cond_img[:, :, ::-1])
            cv2.imwrite(mask_out, cond_mask)

        # logging.info("Saved %d/%d for object %s -> %s", len(selected_data_idx), len(scored), obj_id, obj_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Prepare top-N condition masks/images per object")
    parser.add_argument("--config", default=os.environ.get("EOMT_CONFIG", "configs/dinov2/OC/config.yaml"))
    parser.add_argument("--out_dir", default="prepared_cond")
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--condition_size", type=int, default=224)
    parser.add_argument("--crop_rel_pad", type=float, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    omni_cfg = cfg.get("Omni6dDataset")
    if omni_cfg is None:
        raise KeyError(f"Missing 'Omni6dDataset' in {args.config}")

    data_root = omni_cfg["data_root"]
    model_meta_root = omni_cfg["model_meta_root"]
    scene_names = omni_cfg.get("scene_name", ["ikea", "scannet++"])
    condition_size = int(args.condition_size) 

    index = build_omni6d_index(
        data_root=data_root,
        model_meta_root=model_meta_root,
        scene_names=scene_names,
    )

    logging.info("Omni6D instance entries: %d", len(index["total_inst"]))
    save_topn_per_object(
        index=index,
        out_dir=args.out_dir,
        top_n=args.top_n,
        min_area=args.min_area,
        condition_size=condition_size,
        crop_rel_pad=args.crop_rel_pad,
    )
    