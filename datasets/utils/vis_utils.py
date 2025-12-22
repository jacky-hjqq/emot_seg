import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
import cv2

def _to_uint8_rgb(img):
    """
    支持 PIL.Image / np.ndarray / torch.Tensor
    返回 np.uint8 (H,W,3)
    """
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:  # 灰度
            arr = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[2] in (1,3):
            arr = img if img.shape[2]==3 else np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape[0] in (1,3) and img.shape[2] not in (1,3):
            # 有人可能给 (C,H,W)
            arr = np.transpose(img, (1,2,0))
            if arr.shape[2]==1:
                arr = np.repeat(arr, 3, axis=2)
        else:
            raise ValueError(f"不支持的 numpy 形状: {img.shape}")
    elif torch.is_tensor(img):
        if img.ndim == 3:  # (C,H,W) 或 (H,W,C)
            if img.shape[0] in (1,3):
                arr = img.detach().cpu().numpy().transpose(1,2,0)
                if arr.shape[2]==1:
                    arr = np.repeat(arr, 3, axis=2)
            elif img.shape[2] in (1,3):
                arr = img.detach().cpu().numpy()
                if arr.shape[2]==1:
                    arr = np.repeat(arr, 3, axis=2)
            else:
                raise ValueError(f"不支持的 torch 形状: {tuple(img.shape)}")
        elif img.ndim == 2:  # (H,W)
            arr = img.detach().cpu().numpy()
            arr = np.stack([arr]*3, axis=-1)
        else:
            raise ValueError(f"不支持的 torch 形状: {tuple(img.shape)}")
    else:
        raise TypeError("img 需要是 PIL.Image / numpy.ndarray / torch.Tensor")

    # 归一到 uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).round()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def inv_transform(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Inverse transformation for images that were processed with 
    torchvision.transforms.ToTensor() and Normalize(mean, std).

    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W), already normalized.
        mean (tuple of float): Mean used in the Normalize transform.
        std (tuple of float): Std used in the Normalize transform.

    Returns:
        numpy.ndarray: De-normalized image as uint8 array of shape (H, W, 3).
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    else:
        raise TypeError("Input must be a torch.Tensor")

    # De-normalize
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    # Clamp to [0,1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy (H, W, C), uint8
    img = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return img

def visualize_targets_and_templates(
    img: np.ndarray,
    targets: dict,
    cond_imgs=None,
    include_bg=False,
    bg_color=(0,0,0),
    thumb_size=128,     # Width/Height of the template thumbnail (pixels)
    ncols=8,            # Max columns per row
    tile_pad=8,         # Spacing between templates
    header_h=28,        # Height reserved above the template for "Color Swatch + Text"
    font: ImageFont.ImageFont | None = None,
):
    """
    Generates an overview image:
    [ Left: Original Image | Right: Colored Mask from Targets ] + [ Bottom: Thumbnail grid of cond_imgs (Optional) ]
    
    Args:
        img: (H, W, 3) The original image in RGB format.
        targets: Dictionary containing ground truth/predictions:
            - "masks": (N, H, W) binary masks (0/1 or Boolean).
            - "labels": (N,) class IDs corresponding to the masks.
        cond_imgs: List/Tensor of condition images. The index `i` in cond_imgs corresponds 
                   to the label `i` in targets.
        include_bg: If True, fills the background of the mask visualization with `bg_color`.
    
    Returns:
        np.uint8 (H_total, W_total, 3) visualization image.
    """

    # --- Internal Helper: Ensure conversion to (H,W,3) uint8 ---
    def _to_uint8_rgb(im):
        # Handle PyTorch Tensors
        if torch.is_tensor(im):
            im = im.detach().cpu().numpy()
        
        # Handle NumPy Arrays
        if isinstance(im, np.ndarray):
            # Case 1: Grayscale (H, W) -> Convert to RGB (H, W, 3)
            if im.ndim == 2:
                im = np.stack([im]*3, axis=-1)
            # Case 2: Channel-first (C, H, W) -> Convert to (H, W, C)
            elif im.ndim == 3 and im.shape[0] in [1,3]: 
                im = np.transpose(im, (1,2,0))
            
            # Normalize and Cast
            if im.dtype == np.uint8: return im
            if im.max() <= 1.0: return (im * 255).astype(np.uint8)
            return im.astype(np.uint8)
        
        # Handle PIL Images
        if isinstance(im, Image.Image):
            return np.array(im.convert("RGB"))
        return np.array(im)

    # --- 1. Normalize img ---
    img_np = _to_uint8_rgb(img)
    H, W = img_np.shape[:2]

    # --- 2. Color Generator ---
    # Use matplotlib's tab20 to get consistent color based on index
    cmap = cm.get_cmap("tab20", 20)

    def get_color_for_index(idx):
        idx = int(idx)
        rgba = cmap(idx % 20)[:3]
        return (np.array(rgba) * 255).astype(np.uint8)

    # --- 3. Generate Colored Mask from Targets (Right Side) ---
    # Initialize mask canvas (default black)
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    if include_bg:
        color_mask[:] = np.array(bg_color, dtype=np.uint8)

    # Parse and draw targets
    if targets is not None and "masks" in targets and len(targets["masks"]) > 0:
        # Ensure data is on CPU numpy
        t_masks = targets["masks"]
        t_labels = targets["labels"]
        
        if torch.is_tensor(t_masks):
            t_masks = t_masks.detach().cpu().numpy()
        if torch.is_tensor(t_labels):
            t_labels = t_labels.detach().cpu().numpy()
            
        # Iterate through each instance and draw
        # Note: If masks overlap, the later ones in the list will overwrite earlier ones.
        for mask_i, label_i in zip(t_masks, t_labels):
            # mask_i should be (H, W) binary/bool
            bool_mask = mask_i > 0
            
            # Get color corresponding to the class label
            color = get_color_for_index(label_i)
            
            # Paint the mask region
            color_mask[bool_mask] = color

    # Concatenate: Original Image | Colored Mask
    vis_top = np.concatenate([img_np, color_mask], axis=1).astype(np.uint8)

    # --- 4. If no cond_imgs, return top part immediately ---
    if cond_imgs is None:
        return vis_top

    # --- 5. Normalize cond_imgs -> List[np.uint8 (h,w,3)] ---
    cond_list = []
    if torch.is_tensor(cond_imgs):
        if cond_imgs.ndim == 4: 
            K = cond_imgs.shape[0]
            if cond_imgs.shape[1] in (1,3): # (K,C,H,W)
                cond_list = [ _to_uint8_rgb(cond_imgs[k]) for k in range(K) ]
            else: # (K,H,W,C)
                cond_list = [ _to_uint8_rgb(cond_imgs[k]) for k in range(K) ]
        else: # Single (C,H,W)
            if cond_imgs.ndim == 3: cond_list = [_to_uint8_rgb(cond_imgs)]
    elif isinstance(cond_imgs, np.ndarray):
        if cond_imgs.ndim == 4:
            K = cond_imgs.shape[0]
            if cond_imgs.shape[3] in (1,3): cond_list = [ _to_uint8_rgb(cond_imgs[k]) for k in range(K) ]
            elif cond_imgs.shape[1] in (1,3): cond_list = [ _to_uint8_rgb(np.transpose(cond_imgs[k], (1,2,0))) for k in range(K) ]
        elif cond_imgs.ndim == 3: cond_list = [_to_uint8_rgb(cond_imgs)]
    elif isinstance(cond_imgs, (list, tuple)):
        cond_list = [ _to_uint8_rgb(x) for x in cond_imgs ]
    
    if len(cond_list) == 0:
        return vis_top

    # --- 6. Generate Thumbnail Grid for Templates ---
    tiles = []
    try:
        font = font or ImageFont.load_default()
    except Exception:
        font = None

    for idx, img_i in enumerate(cond_list):
        # Create thumbnail
        pil_img = Image.fromarray(img_i).convert("RGB").resize((thumb_size, thumb_size), Image.BILINEAR)

        # Create tile canvas (including header area)
        tile = Image.new("RGB", (thumb_size, header_h + thumb_size), (255,255,255))
        draw = ImageDraw.Draw(tile)

        # Color Swatch: Use consistent color logic with the Targets
        col = tuple(map(int, get_color_for_index(idx)))
        
        swatch_w, swatch_h = 16, 16
        swatch_pad = 6
        draw.rectangle(
            [swatch_pad, (header_h - swatch_h)//2, swatch_pad+swatch_w, (header_h + swatch_h)//2], 
            fill=col, outline=(0,0,0)
        )

        # Text: Display Index
        text = f"Idx: {idx}"
        text_x = swatch_pad + swatch_w + 6
        try:
            bbox = font.getbbox(text)
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_h = 10 
            
        text_y = (header_h - text_h) // 2
        draw.text((text_x, max(0, text_y)), text, fill=(0,0,0), font=font)

        # Paste image onto tile
        tile.paste(pil_img, (0, header_h))
        tiles.append(np.array(tile, dtype=np.uint8))

    # --- 7. Grid Layout ---
    n = len(tiles)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols

    tile_h, tile_w = tiles[0].shape[0], tiles[0].shape[1]
    grid_h = nrows * tile_h + (nrows - 1) * tile_pad
    grid_w = ncols * tile_w + (ncols - 1) * tile_pad
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for idx, t in enumerate(tiles):
        r = idx // ncols
        c = idx % ncols
        y0 = r * (tile_h + tile_pad)
        x0 = c * (tile_w + tile_pad)
        grid[y0:y0+tile_h, x0:x0+tile_w] = t

    # --- 8. Final Concatenation: Top (vis_top) + Bottom (grid) ---
    pad_h = 12
    pad = np.full((pad_h, max(vis_top.shape[1], grid.shape[1]), 3), 255, dtype=np.uint8)

    # Helper function to center pad width
    def center_pad(a, target_w):
        if a.shape[1] >= target_w: return a
        left = (target_w - a.shape[1]) // 2
        right = target_w - a.shape[1] - left
        return np.pad(a, ((0,0),(left,right),(0,0)), mode="constant", constant_values=255)

    target_w = max(vis_top.shape[1], grid.shape[1])
    vis_top_c = center_pad(vis_top, target_w)
    grid_c = center_pad(grid, target_w)

    vis = np.concatenate([vis_top_c, pad, grid_c], axis=0).astype(np.uint8)
    return vis

def visualize_image_mask_and_templates(
    img: np.ndarray,
    mask,
    cond_imgs=None,
    include_bg=False,
    bg_color=(0,0,0),
    thumb_size=128,     # Width/Height of the template thumbnail (pixels)
    ncols=8,            # Max columns per row
    tile_pad=8,         # Spacing between templates
    header_h=28,        # Height reserved above the template for "Color Swatch + Text"
    font: ImageFont.ImageFont | None = None,
):
    """
    Generates an overview image:
    [ Left: Original Image | Right: Colored Mask ] + [ Bottom: Thumbnail grid of cond_imgs with indices (Optional) ]
    
    Args:
        img (np.ndarray): The original image in RGB format. 
                          Expected shape: (H, W, 3) or (H, W).
        mask (np.ndarray | torch.Tensor | Image): The segmentation mask (H, W).
    
    Current Logic: 
    The value `i` (>=0) in the mask directly corresponds to the i-th image in `cond_imgs`.
    -1 represents the background.

    Returns: np.uint8 (H_total, W_total, 3)
    """

    # --- Internal Helper: Ensure conversion to (H,W,3) uint8 ---
    def _to_uint8_rgb(im):
        # Handle PyTorch Tensors
        if torch.is_tensor(im):
            im = im.detach().cpu().numpy()
        
        # Handle NumPy Arrays
        if isinstance(im, np.ndarray):
            # Case 1: Grayscale (H, W) -> Convert to RGB (H, W, 3)
            if im.ndim == 2:
                im = np.stack([im]*3, axis=-1)
            # Case 2: Channel-first (C, H, W) -> Convert to (H, W, C)
            # Heuristic: if first dim is 1 or 3, assume it's channel dim
            elif im.ndim == 3 and im.shape[0] in [1,3]: 
                im = np.transpose(im, (1,2,0))
            
            # Normalize and Cast
            if im.dtype == np.uint8: 
                return im
            # If float in [0, 1], scale to 255
            if im.max() <= 1.0: 
                return (im * 255).astype(np.uint8)
            return im.astype(np.uint8)
        
        # Handle PIL Images
        if isinstance(im, Image.Image):
            return np.array(im.convert("RGB"))
        
        return np.array(im)

    # --- 1. Normalize img / mask ---
    # Since img is assumed to be numpy, this helper handles it perfectly.
    img_np  = _to_uint8_rgb(img)
    
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    elif torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        raise TypeError("mask must be PIL, numpy array, or torch tensor")

    # Ensure mask is 2D (H, W)
    if mask_np.ndim != 2:
        # If shape is (1, H, W), attempt to squeeze
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np.squeeze(0)
        else:
            raise ValueError(f"mask should be (H,W), current: {mask_np.shape}")
            
    H, W = mask_np.shape
    # Check consistency between image and mask dimensions
    if img_np.shape[:2] != (H, W):
        raise ValueError(f"img and mask size mismatch: img={img_np.shape[:2]}, mask={(H,W)}")

    # --- 2. Color Generator ---
    # Use matplotlib's tab20 to get color based on index
    cmap = cm.get_cmap("tab20", 20)

    def get_color_for_index(idx):
        """Returns RGB color (np.array) directly based on index."""
        idx = int(idx)
        # Negative values (like -1) are usually background; handled externally if needed.
        rgba = cmap(idx % 20)[:3]
        return (np.array(rgba) * 255).astype(np.uint8)

    # --- 3. Generate Colored Mask (Right Side) ---
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    # If background is included and mask contains -1 (assuming -1 is background)
    if include_bg:
        color_mask[mask_np == -1] = np.array(bg_color, dtype=np.uint8)

    # Get all object indices appearing in the mask (filtering out -1)
    unique_indices = np.unique(mask_np)
    unique_indices = unique_indices[unique_indices >= 0] 

    # Color the mask
    for idx in unique_indices:
        color_mask[mask_np == idx] = get_color_for_index(idx)

    # Concatenate: Original Image | Colored Mask
    vis_top = np.concatenate([img_np, color_mask], axis=1).astype(np.uint8)

    # --- 4. If no cond_imgs, return top part immediately ---
    if cond_imgs is None:
        return vis_top

    # --- 5. Normalize cond_imgs -> List[np.uint8 (h,w,3)] ---
    # Supports (K,C,H,W) tensor, (K,H,W,3) numpy, or list
    cond_list = []
    if torch.is_tensor(cond_imgs):
        if cond_imgs.ndim == 4: 
            K = cond_imgs.shape[0]
            # Check channel dimension
            if cond_imgs.shape[1] in (1,3): # (K,C,H,W)
                cond_list = [ _to_uint8_rgb(cond_imgs[k]) for k in range(K) ]
            else: # (K,H,W,C)
                cond_list = [ _to_uint8_rgb(cond_imgs[k]) for k in range(K) ]
        else:
             # Handle single image case (C,H,W) -> list of 1
            if cond_imgs.ndim == 3:
                cond_list = [_to_uint8_rgb(cond_imgs)]
    elif isinstance(cond_imgs, np.ndarray):
        if cond_imgs.ndim == 4:
            K = cond_imgs.shape[0]
            if cond_imgs.shape[3] in (1,3): # (K,H,W,C)
                cond_list = [ _to_uint8_rgb(cond_imgs[k]) for k in range(K) ]
            elif cond_imgs.shape[1] in (1,3): # (K,C,H,W)
                cond_list = [ _to_uint8_rgb(np.transpose(cond_imgs[k], (1,2,0))) for k in range(K) ]
        elif cond_imgs.ndim == 3: # Single image
             cond_list = [_to_uint8_rgb(cond_imgs)]
    elif isinstance(cond_imgs, (list, tuple)):
        cond_list = [ _to_uint8_rgb(x) for x in cond_imgs ]
    
    if len(cond_list) == 0:
        return vis_top

    # --- 6. Generate Thumbnail Grid for Templates ---
    tiles = []
    try:
        font = font or ImageFont.load_default()
    except Exception:
        font = None

    # Iterate through cond_list; the index represents the ID
    for idx, img_i in enumerate(cond_list):
        # Create thumbnail
        pil_img = Image.fromarray(img_i)
        pil_img = pil_img.convert("RGB")
        pil_img = pil_img.resize((thumb_size, thumb_size), Image.BILINEAR)

        # Create tile canvas (including header area)
        tile = Image.new("RGB", (thumb_size, header_h + thumb_size), (255,255,255))
        draw = ImageDraw.Draw(tile)

        # Color Swatch: Use consistent color logic with the Mask
        col = tuple(map(int, get_color_for_index(idx)))
        
        swatch_w, swatch_h = 16, 16
        swatch_pad = 6
        # Draw swatch
        draw.rectangle(
            [swatch_pad, (header_h - swatch_h)//2, swatch_pad+swatch_w, (header_h + swatch_h)//2], 
            fill=col, 
            outline=(0,0,0)
        )

        # Text: Display Index
        text = f"Idx: {idx}"
        text_x = swatch_pad + swatch_w + 6
        
        # Calculate centered text position
        try:
            bbox = font.getbbox(text)
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_h = 10 
            
        text_y = (header_h - text_h) // 2
        draw.text((text_x, max(0, text_y)), text, fill=(0,0,0), font=font)

        # Paste image onto tile
        tile.paste(pil_img, (0, header_h))
        tiles.append(np.array(tile, dtype=np.uint8))

    # --- 7. Grid Layout ---
    n = len(tiles)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols

    tile_h, tile_w = tiles[0].shape[0], tiles[0].shape[1]
    grid_h = nrows * tile_h + (nrows - 1) * tile_pad
    grid_w = ncols * tile_w + (ncols - 1) * tile_pad
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for idx, t in enumerate(tiles):
        r = idx // ncols
        c = idx % ncols
        y0 = r * (tile_h + tile_pad)
        x0 = c * (tile_w + tile_pad)
        grid[y0:y0+tile_h, x0:x0+tile_w] = t

    # --- 8. Final Concatenation: Top (vis_top) + Bottom (grid) ---
    pad_h = 12
    pad = np.full((pad_h, max(vis_top.shape[1], grid.shape[1]), 3), 255, dtype=np.uint8)

    # Helper function to center pad width
    def center_pad(a, target_w):
        if a.shape[1] >= target_w:
            return a
        left = (target_w - a.shape[1]) // 2
        right = target_w - a.shape[1] - left
        return np.pad(a, ((0,0),(left,right),(0,0)), mode="constant", constant_values=255)

    target_w = max(vis_top.shape[1], grid.shape[1])
    vis_top_c = center_pad(vis_top, target_w)
    grid_c = center_pad(grid, target_w)

    vis = np.concatenate([vis_top_c, pad, grid_c], axis=0).astype(np.uint8)
    return vis

def visualize_bbox(image, bbox, score, obj_id, color=(255, 0, 0), thickness=2, font_scale=0.5):
    """
    image: np.array, HWC (h, w, 3), RGB
    bbox: [x, y, w, h]
    score: float
    obj_id: int
    """
    img_copy = image.copy()

    if img_copy.dtype != np.uint8:
        img_copy = (img_copy * 255).clip(0, 255).astype(np.uint8)

    x, y, w, h = bbox
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

    # 画矩形框
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

    # 文本标签：id + score
    label = f"Detect_id:{obj_id} & bbox_score: {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(img_copy, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)  # 背景条
    cv2.putText(img_copy, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return img_copy

def visualize_instance_mask(img, instance_mask):
    """
    Visualizes image and instance mask side-by-side using OpenCV.
    
    Args:
        img (torch.Tensor): Shape (3, H, W). Assumes RGB.
        instance_mask (torch.Tensor): Shape (H, W). -1 is background.
        
    Returns:
        np.ndarray: The combined visualization (H, 2*W, 3) in BGR format (uint8).
    """
    # 1. Process Original Image
    # Convert to numpy and transpose to (H, W, C)
    img_np = img.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Normalize to 0-255 uint8
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    # 2. Process Instance Mask
    mask_np = instance_mask.cpu().detach().numpy()
    H, W = mask_np.shape
    
    # Initialize empty BGR image (black background)
    mask_colored = np.zeros((H, W, 3), dtype=np.uint8)
    
    unique_ids = np.unique(mask_np)
    
    for uid in unique_ids:
        if uid == -1:
            # Explicitly leave background black (already zeros)
            continue
        
        # Generate random BGR color
        # high=256 ensures vivid colors
        color = np.random.randint(0, 256, size=3).tolist()
        
        # Assign color to the mask pixels
        mask_colored[mask_np == uid] = color

    # 3. Combine Side-by-Side
    # Horizontal concatenation
    vis_result = np.hstack((img_np, mask_colored))
    
    return vis_result