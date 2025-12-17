import numpy as np
import cv2

def get_mask(rgb_array, background="auto", tol=5):
    """
    Generate a foreground mask from an RGB numpy image with a black or white background.
    :param rgb_array: Input image (H, W, 3, RGB)
    :param background: "auto" | "black" | "white"
    :param tol: Tolerance for near-black/near-white values (to handle compression artifacts)
    :return: mask (H, W), foreground=255, background=0
    """
    if background == "auto":
        mean_val = np.mean(rgb_array)
        background = "white" if mean_val > 127 else "black"

    if background == "white":
        # Background is close to white → foreground = non-white pixels
        mask = np.any(rgb_array < 255 - tol, axis=-1).astype(np.uint8) * 255
    else:
        # Background is close to black → foreground = non-black pixels
        mask = np.any(rgb_array > tol, axis=-1).astype(np.uint8) * 255

    return mask

def crop_and_resize(image, mask, size=224, crop_rel_pad=0.2, pad_value=(0,0,0)):
    """
    Crop around the mask region, resize with aspect ratio, and pad to square.
    Both image and mask are transformed consistently.
    
    :param image: RGB numpy array (H, W, C)
    :param mask: binary mask (H, W)
    :param size: output size (square)
    :param crop_rel_pad: relative padding around the bounding box
    :param pad_value: background pad color (for image)
    :return: (out_image, out_mask)
    """
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # No foreground → return padded blank
        out_img = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
        out_mask = np.zeros((size, size), dtype=np.uint8)
        return out_img, out_mask

    # bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # padding
    pad_x = int((x_max - x_min + 1) * crop_rel_pad)
    pad_y = int((y_max - y_min + 1) * crop_rel_pad)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(W, x_max + pad_x)
    y2 = min(H, y_max + pad_y)

    # crop
    cropped_img = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    # scale
    h, w = cropped_img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # pad to square
    out_img = np.full((size, size, image.shape[2]), pad_value, dtype=resized_img.dtype)
    out_mask = np.zeros((size, size), dtype=np.uint8)

    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2

    out_img[y0:y0+new_h, x0:x0+new_w] = resized_img
    out_mask[y0:y0+new_h, x0:x0+new_w] = resized_mask

    return out_img, out_mask