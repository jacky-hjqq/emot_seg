# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict
from torchvision import transforms
import numpy as np
import cv2
import random
from typing import Tuple, Iterable

def get_image_augmentation(
    color_jitter: Optional[Dict[str, float]] = None,
    gray_scale: bool = True,
    gau_blur: bool = False
) -> Optional[transforms.Compose]:
    """Create a composition of image augmentations.

    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "p": 0.9
    }

    # Handle color jitter
    if color_jitter is not None:
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if gray_scale:
        transform_list.append(transforms.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )

    return transforms.Compose(transform_list) if transform_list else None

def random_rotation(img: np.ndarray,
                    mask: np.ndarray,
                    angle_deg: float,
                    expand: bool = True,
                    bg_value=[255, 255, 255]):
    H, W = mask.shape[:2]
    dtype_img = img.dtype
    dtype_mask = mask.dtype

    # 前景二值用于取 bbox
    bin_fg = (mask != 0)
    ys, xs = np.where(bin_fg)
    if len(xs) == 0:  # 无前景
        return img.copy(), mask.copy()

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    img_patch  = img[y0:y1, x0:x1].copy()
    mask_patch = mask[y0:y1, x0:x1].copy()

    # 旋转矩阵（绕 patch 中心）
    h, w = img_patch.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)

    # 计算扩展后的尺寸与平移补偿
    if expand:
        cos = abs(M[0, 0]); sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w/2 - w/2)
        M[1, 2] += (new_h/2 - h/2)
    else:
        new_w, new_h = w, h

    # 边界填充值
    if bg_value is None:
        border_val_img = (255, 255, 255) if img.ndim == 3 else 0
    else:
        border_val_img = tuple(bg_value) if img.ndim == 3 else (bg_value if np.isscalar(bg_value) else 0)

    # 旋转：图像用双线性，mask 用最近邻（保持类别）
    img_rot  = cv2.warpAffine(img_patch,  M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=border_val_img)
    mask_rot = cv2.warpAffine(mask_patch, M, (new_w, new_h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

    # 贴回位置：以原 bbox 中心为锚点
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    px = int(round(cx - new_w / 2))
    py = int(round(cy - new_h / 2))

    # 计算与原图的重叠区域，避免越界
    xA = max(0, px); yA = max(0, py)
    xB = min(W, px + new_w); yB = min(H, py + new_h)
    if xA >= xB or yA >= yB:
        return img.copy(), mask.copy()  # 全在画布外，直接返回

    # 对应到旋转patch的裁剪窗口
    rxA = xA - px; ryA = yA - py
    rxB = rxA + (xB - xA); ryB = ryA + (yB - yA)

    out_img  = img.copy()
    out_mask = mask.copy()

    # ---------- 关键修复：先把旧前景“抠掉”成背景 ----------
    # 仅抠 bbox 内即可（与 mask 提取的范围一致）
    old_fg = (mask[y0:y1, x0:x1] != 0)
    if img.ndim == 3:
        bg = np.array(border_val_img, dtype=out_img.dtype)
        out_img[y0:y1, x0:x1][old_fg] = bg
    else:
        bg = 0 if bg_value is None else (int(bg_value) if np.isscalar(bg_value) else 0)
        out_img[y0:y1, x0:x1][old_fg] = bg
    # 同步清掉旧 mask
    out_mask[y0:y1, x0:x1] = 0
    # -------------------------------------------------------

    # 只在旋转后 mask>0 的位置贴回
    alpha = (mask_rot[ryA:ryB, rxA:rxB] != 0)

    if img.ndim == 3:
        patch  = img_rot[ryA:ryB, rxA:rxB]
        region = out_img[yA:yB, xA:xB]
        region[alpha] = patch[alpha]
        out_img[yA:yB, xA:xB] = region
    else:
        patch  = img_rot[ryA:ryB, rxA:rxB]
        region = out_img[yA:yB, xA:xB]
        region[alpha] = patch[alpha]
        out_img[yA:yB, xA:xB] = region

    # 写回 mask（保持类别）
    region_m = out_mask[yA:yB, xA:xB]
    patch_m  = mask_rot[ryA:ryB, rxA:rxB]
    region_m[alpha] = patch_m[alpha]
    out_mask[yA:yB, xA:xB] = region_m

    return out_img.astype(dtype_img, copy=False), out_mask.astype(dtype_mask, copy=False)

def random_scaled_crop(
    image: np.ndarray,           # HxW 或 HxWxC
    mask: np.ndarray,            # HxW，元素为 cls id（整数）
    crop_scale: float,
    focus_on_objects: bool = True,
    focus_strength: float = 2.0,      # >1 越偏向前景密集区域
    background_ids: Iterable[int] | None = (0, -1),  # 视作背景/忽略的 id
):
    """
    随机按比例裁剪（s*H, s*W），保持纵横比不变；不做 padding；最后 resize 回 (H,W)。
    当 focus_on_objects=True 时，窗口按“前景像素计数^focus_strength”加权采样。

    返回: (aug_image, aug_mask) 与输入同尺寸与dtype
    """
    assert image.shape[:2] == mask.shape[:2]
    H, W = mask.shape[:2]

    crop_h = max(1, int(round(H * crop_scale)))
    crop_w = max(1, int(round(W * crop_scale)))

    # 定义前景
    if background_ids is None:
        fg = (mask != 0)
    else:
        fg = ~np.isin(mask, list(background_ids))

    # 当裁剪尺寸与原图一致时，直接返回
    if crop_h == H and crop_w == W:
        return image.copy(), mask.copy()

    # 在原图内选择 top-left (y0,x0)
    if not focus_on_objects:
        y0 = 0 if H == crop_h else random.randint(0, H - crop_h)
        x0 = 0 if W == crop_w else random.randint(0, W - crop_w)
    else:
        # 用积分图计算每个候选窗口的前景计数（向量化）
        integ = cv2.integral(fg.astype(np.uint8))  # (H+1, W+1)
        hh, ww = H - crop_h + 1, W - crop_w + 1
        if hh <= 0 or ww <= 0:
            # 理论上不会发生（因为 s<=1），保险起见
            y0 = 0
            x0 = 0
        else:
            sums = (integ[crop_h:, crop_w:]
                    - integ[:-crop_h, crop_w:]
                    - integ[crop_h:, :-crop_w]
                    + integ[:-crop_h, :-crop_w]).astype(np.float64)  # [hh, ww]
            if sums.max() <= 0:
                # 没有前景，退化为均匀随机
                y0 = 0 if H == crop_h else random.randint(0, H - crop_h)
                x0 = 0 if W == crop_w else random.randint(0, W - crop_w)
            else:
                weights = np.power(sums + 1e-6, focus_strength)
                probs = (weights / weights.sum()).ravel()
                idx = np.random.choice(probs.size, p=probs)
                y0, x0 = divmod(idx, ww)
                y0, x0 = int(y0), int(x0)

    img_c = image[y0:y0+crop_h, x0:x0+crop_w]
    msk_c = mask[y0:y0+crop_h, x0:x0+crop_w]

    # resize 回原分辨率（图像线性，mask 最近邻）
    img_out = cv2.resize(img_c, (W, H), interpolation=cv2.INTER_LINEAR)
    msk_out = cv2.resize(msk_c, (W, H), interpolation=cv2.INTER_NEAREST)

    # 维护通道与dtype
    if image.ndim == 3 and img_out.ndim == 2:
        img_out = img_out[..., None]
    img_out = img_out.astype(image.dtype, copy=False)
    msk_out = msk_out.astype(mask.dtype, copy=False)

    return img_out, msk_out

def object_centric_crop(img, mask, crop_scale_range=(3.0, 5.0), max_shift_ratio=0.2):
    """
    Args:
        img: np.array (H, W, 3), RGB
        mask: np.array (H, W), int
        crop_scale_range: 控制裁剪区域相对物体 bbox 的放大倍数范围
        max_shift_ratio: 最大偏移比例（相对于 crop 尺寸）

    Returns:
        cropped_img: (H, W, 3)，保持原图比例
        cropped_mask: (H, W)，保持原图比例
    """
    H, W = mask.shape
    aspect = W / H

    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]
    if len(unique_ids) == 0:
        return img, mask

    # 随机选一个类别
    chosen_cls = random.choice(unique_ids)

    # 如果物体mask过小，直接返回
    mask_area = np.sum(mask == chosen_cls)
    if mask_area < 0.001 * H * W:
        return img, mask
    
    # 物体 bbox
    ys, xs = np.where(mask == chosen_cls)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    box_h, box_w = y_max - y_min + 1, x_max - x_min + 1
    cy, cx = int((y_min + y_max) / 2), int((x_min + x_max) / 2)

    # 随机扩展
    scale = random.uniform(*crop_scale_range)
    crop_h = int(box_h * scale)
    crop_w = int(box_w * scale)

    # 调整为原图比例
    if crop_w / crop_h > aspect:
        crop_h = int(crop_w / aspect)
    else:
        crop_w = int(crop_h * aspect)

    # 保证不小于 bbox 尺寸
    crop_h = max(crop_h, box_h)
    crop_w = max(crop_w, box_w)

    # ---- 关键：计算合法的中心偏移范围 ----
    # y 方向
    min_cy = max(y_max - crop_h // 2, 0 + crop_h // 2)
    max_cy = min(y_min + crop_h // 2, H - crop_h // 2)
    # x 方向
    min_cx = max(x_max - crop_w // 2, 0 + crop_w // 2)
    max_cx = min(x_min + crop_w // 2, W - crop_w // 2)

    # 在合法范围内再加一个 max_shift_ratio 的限制
    shift_range_y = int(max_shift_ratio * crop_h)
    shift_range_x = int(max_shift_ratio * crop_w)

    min_cy = max(min_cy, cy - shift_range_y)
    max_cy = min(max_cy, cy + shift_range_y)
    min_cx = max(min_cx, cx - shift_range_x)
    max_cx = min(max_cx, cx + shift_range_x)

    # 随机新中心
    new_cy = random.randint(min_cy, max_cy) if min_cy < max_cy else cy
    new_cx = random.randint(min_cx, max_cx) if min_cx < max_cx else cx

    # 计算 crop 框（限制在边界内）
    y1 = max(0, new_cy - crop_h // 2)
    y2 = min(H, new_cy + crop_h // 2)
    x1 = max(0, new_cx - crop_w // 2)
    x2 = min(W, new_cx + crop_w // 2)

    # 防止空 crop
    if y2 <= y1 or x2 <= x1:
        return img, mask

    # 裁剪
    crop_img = img[y1:y2, x1:x2, :]
    crop_mask = mask[y1:y2, x1:x2]

    # resize 回原图大小 (比例一致，无需 padding)
    cropped_img = cv2.resize(crop_img, (W, H), interpolation=cv2.INTER_LINEAR)
    cropped_mask = cv2.resize(crop_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    return cropped_img, cropped_mask