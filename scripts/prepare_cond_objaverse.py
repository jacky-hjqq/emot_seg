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

def visualize_instance(rgb, mask, inst_id, save_path, alpha=0.5, color=(0, 255, 0)):
    """
    可视化特定 instance id 的 mask 并保存。
    
    参数:
        rgb: 输入的 RGB 图像矩阵 (H, W, 3) 或 (H, W, 4)
        mask: 语义/实例分割 mask (H, W), dtype=uint16
        inst_id: 需要可视化的 instance id
        save_path: 图片保存路径
        alpha: 透明度 (0.0 - 1.0), 越小越透明
        color: 掩膜颜色 (R, G, B), 默认绿色
    """
    
    # 1. 预处理：如果输入是 RGBA (4通道)，只取前3个通道 (RGB)
    if rgb.shape[-1] == 4:
        img_viz = rgb[:, :, :3].copy()
    else:
        img_viz = rgb.copy()
        
    # 确保图像是 uint8 类型以便处理
    if img_viz.dtype != np.uint8:
        img_viz = img_viz.astype(np.uint8)

    # 2. 提取特定的 Instance Mask
    # 创建一个二值掩膜：目标区域为 1，背景为 0
    binary_mask = (mask == inst_id).astype(np.uint8)
    
    # 如果 mask 里没有这个 id，直接保存原图并返回
    if np.sum(binary_mask) == 0:
        print(f"Warning: Instance ID {inst_id} not found in mask.")
        # OpenCV 保存需要 BGR
        cv2.imwrite(save_path, cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR))
        return

    # 3. 应用半透明颜色填充
    # 创建一个纯色的图层
    color_layer = np.zeros_like(img_viz)
    color_layer[:] = color
    
    # 获取需要上色的区域索引
    roi_indices = np.where(binary_mask == 1)
    
    # 在原图上进行线性混合: pixel = img * (1-a) + color * a
    img_viz[roi_indices] = (img_viz[roi_indices] * (1 - alpha) + 
                            color_layer[roi_indices] * alpha).astype(np.uint8)

    # 4. (可选) 添加轮廓线让边界更清晰
    # findContours 需要 uint8 单通道图像
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在图上画轮廓，颜色为白色 (255, 255, 255)，线宽 2
    cv2.drawContours(img_viz, contours, -1, (255, 255, 255), 2)

    # 5. 保存结果
    # 注意：cv2.imwrite 默认接受 BGR 格式，而你的变量名是 rgb，
    # 为了保证保存的颜色正确，我们需要将 RGB 转回 BGR。
    img_bgr = cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    print(f"Saved visualization for ID {inst_id} to {save_path}")

def get_paths(
    *,
    data_root: str,
    data_patch: str,
    scene_id: str,
    scene_name: str,
    render_name: str,
    color_tpath: str,
    mask_tpath: str,
    instance_map_tpath: str,
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
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return paths


def build_index(data_root: str):
    color_tpath = "{data_root}/{data_patch}/{scene_id}/{scene_name}/{render_name}/rgb/rgb_000000.png"
    mask_tpath = "{data_root}/{data_patch}/{scene_id}/{scene_name}/{render_name}/instance_segmentation/instance_segmentation_000000.png"
    instance_map_tpath = "{data_root}/{data_patch}/{scene_id}/{scene_name}/{render_name}/instance_segmentation/instance_segmentation_mapping_000000.json"
    
    total_inst: list[dict] = []
    total_images = 0
    annotation_objects: dict[str, list[dict]] = defaultdict(list)
    data_patches = sorted(os.listdir(os.path.join(data_root)))
    for data_patch in data_patches:
        scene_ids = sorted(os.listdir(os.path.join(data_root, data_patch)))
        for scene_id in scene_ids:
            scene_names = sorted(os.listdir(os.path.join(data_root, data_patch, scene_id)))
            for scene_name in scene_names:
                render_names = sorted(os.listdir(os.path.join(data_root, data_patch, scene_id, scene_name)))
                for render_name in render_names:
                    paths = get_paths(
                        data_root=data_root,
                        data_patch=data_patch,
                        scene_id=scene_id,
                        scene_name=scene_name,
                        render_name=render_name,
                        color_tpath=color_tpath,
                        mask_tpath=mask_tpath,
                        instance_map_tpath=instance_map_tpath,
                    )
                    if paths is None:
                        continue
                    total_images += 1

                    # load rgb and mask
                    # # visualize instance
                    # mask = cv2.imread(paths["mask"], cv2.IMREAD_UNCHANGED).astype(np.uint8)
                    # mask_ids = np.unique(mask)
                    # rgb = cv2.imread(paths["color"])[:,:,::-1]  # BGR to RGB
                    # visualize_instance(rgb, mask, inst_id=4, save_path="instance_viz.png", alpha=0.5)

                    # load instance map 
                    with open(paths["instance_map"], "r") as f:
                        instance_map = json.load(f)

                    for idx, key in enumerate(instance_map):
                        value = instance_map[key]
                        value = value.lower()
                        # skip background and unlabeled
                        if value in ["background", "unlabelled"]:
                            continue
                        # skip if inst not object
                        value_type = value.split("/")[2]
                        if value_type != "objects":
                            continue
                        obj_name = value.split("/")[-2]
                        data_idx = len(total_inst)
                        total_inst.append(
                            {
                                "data_idx": data_idx,
                                "image_path": paths["color"],
                                "mask_path": paths["mask"],
                                "inst_id": int(key),
                                "obj_name": obj_name,
                            }
                        )
                        annotation_objects[obj_name].append({"data_idx": data_idx})

        # break

    return {
        "total_images": total_images,
        "total_inst": total_inst,
        "annotation_objects": annotation_objects,
    }


def _sanitize_dir_name(name: str) -> str:
    name = str(name)
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    name = name.strip().strip(".")
    return name or "unknown"

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
            shutil.rmtree(obj_dir) if os.path.exists(obj_dir) else None
        os.makedirs(obj_dir, exist_ok=True)

        # 1) Score each instance by mask_area
        scored: list[int, int] = []  # (mask_area, data_idx)
        for entry in entries:
            data_idx = int(entry["data_idx"])
            data = total_inst[data_idx]
            inst_id = int(data["inst_id"])
            mask = cv2.imread(data["mask_path"], cv2.IMREAD_UNCHANGED)
            inst_mask = (mask == inst_id)
            mask_area = int(inst_mask.sum())
            if mask_area < min_area:
                continue
            scored.append((mask_area, data_idx))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0]) # based on the mask_area, ascending
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

            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            inst_mask = (mask == inst_id)
            mask_u8 = (inst_mask.astype(np.uint8) * 255)

            # Resize + crop to match training templates, then apply the mask.
            cond_img, cond_mask = crop_and_resize(
                img_rgb,
                mask_u8,
                size=int(condition_size),
                crop_rel_pad=float(crop_rel_pad),
            )
            cond_mask = cond_mask.astype(np.uint8)
            cond_img = cond_img * cond_mask[:, :, None].astype(bool)

            # Use the richest available identifier, but stay robust across datasets.
            if all(k in data for k in ("data_patch", "scene_id", "scene_name", "frame_id")):
                stem = f"{rank:03d}_{data['data_patch']}_{data['scene_id']}_{data['scene_name']}_{data['frame_id']}"
            elif all(k in data for k in ("scene_id", "frame_id")):
                stem = f"{rank:03d}_{data['scene_id']}_{data['frame_id']}"
            else:
                base = os.path.splitext(os.path.basename(image_path))[0]
                stem = f"{rank:03d}_{base}_inst{inst_id}_idx{data_idx}"
            img_out = os.path.join(obj_dir, stem + "_image.png")
            mask_out = os.path.join(obj_dir, stem + "_mask.png")

            cv2.imwrite(img_out, cond_img[:, :, ::-1])
            cv2.imwrite(mask_out, cond_mask)

        # logging.info("Saved %d/%d for object %s -> %s", len(selected_data_idx), len(scored), obj_id, obj_dir)


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Prepare top-N condition masks/images per object")
    parser.add_argument("--config", default=os.environ.get("EOMT_CONFIG", "configs/dinov2/OC/config.yaml"))
    parser.add_argument("--out_dir", default="prepared_cond_objaverse")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--min_area", type=int, default=1000)
    parser.add_argument("--condition_size", type=int, default=224)
    parser.add_argument("--crop_rel_pad", type=float, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    objaverse_cfg = cfg.get("ObjaverseDataset")
    if objaverse_cfg is None:
        raise KeyError(f"Missing 'ObjaverseDataset' in {args.config}")

    data_root = objaverse_cfg["data_root"]
    condition_size = int(args.condition_size) 

    start = time.time()
    index = build_index(data_root=data_root)
    logging.info("Objaverse index built in %.1f sec", time.time() - start)
    logging.info("Objaverse instance entries: %d", len(index["total_inst"]))
    logging.info("Objaverse total images: %d", int(index["total_images"]))
    save_topn_per_object(
        index=index,
        out_dir=args.out_dir,
        top_n=args.top_n,
        min_area=args.min_area,
        condition_size=condition_size,
        crop_rel_pad=args.crop_rel_pad,
    )
    