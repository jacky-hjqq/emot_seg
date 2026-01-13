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
import sys
sys.path.append(".")
from datasets.utils.img_utils import crop_and_resize
from datasets.utils.vis_utils import *

def build_ycbv_index(*, data_root: str):
    """Build an instance index for BOP YCBV-style datasets.

    Expects:
      - {data_root}/{scene_id}/rgb/{frame_id:06d}.png
      - {data_root}/{scene_id}/mask_visib/{frame_id:06d}_{inst_id:06d}.png
      - {data_root}/{scene_id}/scene_gt.json
    """
    total_inst: list[dict] = []
    annotation_objects: dict[str, list[dict]] = defaultdict(list)

    scenes = sorted(os.listdir(os.path.join(data_root)))
    for scene in scenes:
        scene_dir = os.path.join(data_root, scene)
        if not os.path.isdir(scene_dir):
            continue

        scene_rgb_dir = os.path.join(scene_dir, "rgb")
        scene_gt_path = os.path.join(scene_dir, "scene_gt.json")
        if not (os.path.isdir(scene_rgb_dir) and os.path.exists(scene_gt_path)):
            continue

        with open(scene_gt_path, "r") as f:
            scene_gt = json.load(f)

        frames = sorted([f for f in os.listdir(scene_rgb_dir) if f.endswith(".jpg")])
        for frame in frames:
            frame_id = int(frame.split(".")[0])
            frame_gt = scene_gt.get(str(frame_id), [])
            frame_image_path = os.path.join(scene_rgb_dir, frame)

            for inst_id, item in enumerate(frame_gt):
                obj_id = int(item["obj_id"])
                obj_key = f"obj_{obj_id:06d}"
                obj_mask_path = os.path.join(
                    scene_dir,
                    "mask_visib",
                    f"{frame_id:06d}_{inst_id:06d}.png",
                )
                if not (os.path.exists(frame_image_path) and os.path.exists(obj_mask_path)):
                    continue

                data_idx = len(total_inst)
                total_inst.append(
                    {
                        "data_idx": data_idx,
                        "image_path": frame_image_path,
                        "mask_path": obj_mask_path,
                        "inst_id": inst_id,
                        "obj_name": obj_key,
                        "obj_id": obj_id,
                        "scene_id": int(scene),
                        "frame_id": frame_id,
                    }
                )
                annotation_objects[obj_key].append({"data_idx": data_idx})

    return {
        "total_inst": total_inst,
        "annotation_objects": annotation_objects,
    }


def _sanitize_dir_name(name: str) -> str:
    name = str(name)
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    name = name.strip().strip(".")
    return name or "unknown"


def _load_instance_mask(mask_path: str, inst_id: int) -> np.ndarray:
    ext = os.path.splitext(mask_path)[1].lower()
    if ext == ".exr":
        # Omni6D masks are stored in EXR; historically we used channel 0 * 255 -> uint8 ids.
        m = imread(mask_path)
        if m.ndim == 3:
            m = m[:, :, 0]
        mask_u8 = (m * 255).astype(np.uint8)
        return mask_u8 == np.uint8(inst_id)

    # BOP-style per-instance visible mask PNGs.
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8) > 0


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

            stem = f"{rank:03d}_{data['scene_id']}_{data['frame_id']}"
            img_out = os.path.join(obj_dir, stem + "_image.png")
            mask_out = os.path.join(obj_dir, stem + "_mask.png")

            cv2.imwrite(img_out, cond_img[:, :, ::-1])
            cv2.imwrite(mask_out, cond_mask)

        # logging.info("Saved %d/%d for object %s -> %s", len(selected_data_idx), len(scored), obj_id, obj_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Prepare top-N condition masks/images per object")
    parser.add_argument("--config", default=os.environ.get("EOMT_CONFIG", "configs/dinov2/OC/config.yaml"))
    parser.add_argument("--out_dir", default="prepared_cond_ycbv")
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--condition_size", type=int, default=224)
    parser.add_argument("--crop_rel_pad", type=float, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ycbv_cfg = cfg.get("YCBV_Train_Dataset")
    if ycbv_cfg is None:
        raise KeyError(f"Missing 'YCBV_Train_Dataset' in {args.config}")

    data_root = ycbv_cfg["data_root"]
    condition_size = int(args.condition_size)

    index = build_ycbv_index(data_root=data_root)

    logging.info("YCBV instance entries: %d", len(index["total_inst"]))
    save_topn_per_object(
        index=index,
        out_dir=args.out_dir,
        top_n=args.top_n,
        min_area=args.min_area,
        condition_size=condition_size,
        crop_rel_pad=args.crop_rel_pad,
    )
    