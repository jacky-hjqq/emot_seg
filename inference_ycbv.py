import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import yaml
import importlib
import numpy as np  
import torch.nn.functional as F
from lightning import seed_everything
from datasets.ycbv_dataset import YCBVSegmentation
from datasets.utils.vis_utils import visualize_image_mask_and_templates

def mask_to_rle_fast(binary_mask: np.ndarray):
    """
    Convert binary mask to uncompressed RLE.
    Much faster than Python loop.

    Args:
        binary_mask: numpy array [H, W] with {0,1}
    Returns:
        RLE: {"size": [H, W], "counts": [list of run lengths]}
    """
    h, w = binary_mask.shape
    # flatten in Fortran order
    flat = binary_mask.ravel(order="F").astype(np.uint8)

    # pad with 0 at start and end to catch transitions
    padded = np.pad(flat, (1, 1), mode="constant", constant_values=0)

    # find where values change
    changes = np.where(padded[1:] != padded[:-1])[0]

    # run lengths are diffs between change indices
    counts = np.diff(np.concatenate(([0], changes, [len(flat)]))).tolist()

    return {"size": [h, w], "counts": counts}

vis_bbox_enable = False
vis_mask_enable = False
os.makedirs("output", exist_ok=True)
dataset_name = "ycbv"

# Instantiate model
seed_everything(0, verbose=False)
device = 0  
img_size = (480, 640)
state_dict_path = "eomt_seg/version-4/checkpoints/epoch=11-step=96000.ckpt"
config_path = "configs/dinov2/OC/inference.yaml" 
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load encoder
encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
encoder = encoder_cls(**encoder_cfg.get("init_args", {}))
# Load cond_encoder
cond_encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["cond_encoder"]
cond_encoder_module_name, cond_encoder_class_name = cond_encoder_cfg["class_path"].rsplit(".", 1)
cond_encoder_cls = getattr(importlib.import_module(cond_encoder_module_name), cond_encoder_class_name)
cond_encoder = cond_encoder_cls(**cond_encoder_cfg.get("init_args", {}))

# Load network
network_cfg = config["model"]["init_args"]["network"]
network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k not in ["encoder", "cond_encoder"]}
network = network_cls(
    masked_attn_enabled=False, # still enable masked_attn for inference
    encoder=encoder,
    cond_encoder=cond_encoder,
    **network_kwargs,
)

# Load Lightning module
lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
model = (
    lit_cls(
        img_size=img_size,
        network=network,
        **model_kwargs,
    ).eval().to(device)
)
# Load checkpoint - it contains the full Lightning module state
checkpoint = torch.load(
    state_dict_path, map_location=f"cuda:{device}", weights_only=False
)
# Extract the state_dict from the checkpoint
state_dict = checkpoint["state_dict"]
model.load_state_dict(state_dict, strict=False)

# Initialize the dataset
config_path = "configs/dinov2/OC/config.yaml"
with open(config_path, 'r') as f:
    data_config = yaml.safe_load(f)
ycbv_config = data_config['YCBVDataset']
dataset = YCBVSegmentation(ycbv_config)

# Inference
seg_res = []
for idx in tqdm(range(len(dataset))):
    batch_data = dataset[idx]
    scene_id = batch_data["scene_id"]
    image_id = batch_data["image_id"]
    img = batch_data["img"] # (3,h,w)
    cond_imgs = batch_data["cond_imgs"] 
    cond_masks = batch_data["cond_masks"]

    with torch.no_grad():
        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(img.unsqueeze(0).to(device))
        input_img_size = [img.shape[-2:]]
        cond_imgs = [cond_imgs.to(device)]   
        cond_masks = [cond_masks.to(device)]
        start = time.time()
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs, cond_imgs, cond_masks)
        # infer_time = time.time() - start
        # visualization
        vis_mask_logits = F.interpolate(mask_logits_per_layer[-1], img_size, mode="bilinear")
        vis_mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
            vis_mask_logits, input_img_size
        )
        pred = model.to_per_pixel_preds_instance(
            vis_mask_logits[0],
            class_logits_per_layer[-1][0],
            model.mask_thresh,
            model.overlap_thresh,
        )
        # infer_time = time.time() - start
        # print(f"Inference time for image {idx}: {infer_time:.3f} seconds")
        semantic_mask, instance_mask = pred[..., 0], pred[..., 1]

        pred_vis = visualize_image_mask_and_templates(
            img=img, 
            mask=semantic_mask,
            cond_imgs=cond_imgs[0],
        )

        save_path = os.path.join(f"output/{dataset_name}_infer_vis", f"{idx:06d}_pred.png")
        os.makedirs(f"output/{dataset_name}_infer_vis", exist_ok=True)
        plt.imsave(save_path, pred_vis.astype(np.uint8))

#     # save the results
#     for inst_mask in instances_by_image[0]:
#         cls_idx = inst_mask['class_id']
#         obj_cond_image = cond_imgs[0][cls_idx]
#         obj_id = batch_data['cond_obj_ids'][cls_idx].item()
#         bbox = inst_mask['bbox'] # xywh
#         bbox_score = inst_mask['score']
#         mask = inst_mask['mask']
#         # convert mask to coco rle
#         rle = mask_to_rle_fast(mask)
#         seg_res.append({
#             "scene_id": scene_id,
#             "image_id": image_id,
#             "category_id": obj_id,
#             "bbox": bbox,
#             "segmentation": rle,    
#             "score": bbox_score,
#             "time": infer_time,
#         })

# # save the seg_res as a json file
# with open(f"output/{dataset_name}_seg_results.json", "w") as f:
#     json.dump(seg_res, f)
    
