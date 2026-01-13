import os
import torch
from tqdm import tqdm
import yaml
import importlib
import numpy as np  
import torch.nn.functional as F
from lightning import seed_everything
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from datasets.omni6d_dataset import Omni6DSegmentation
from datasets.utils.vis_utils import visualize_image_mask_and_templates, visualize_targets_and_templates

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

os.makedirs("output", exist_ok=True)
dataset_name = "omni6d"

# Instantiate model
seed_everything(0, verbose=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _to_device(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x

img_size = (480, 640)
state_dict_path = 'eomt_seg/7338do0e/checkpoints/epoch=2-step=24000.ckpt'
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
    masked_attn_enabled=True, # still enable masked_attn for inference
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
    state_dict_path, map_location=device, weights_only=False
)
# Extract the state_dict from the checkpoint
state_dict = checkpoint["state_dict"]
model.load_state_dict(state_dict, strict=False)

# Initialize the dataset
config_path = "configs/dinov2/OC/config.yaml"
with open(config_path, 'r') as f:
    data_config = yaml.safe_load(f)
omni6d_config = data_config['Omni6dDataset']
dataset = Omni6DSegmentation(omni6d_config, split="test")

# Inference
val_loss_sum = 0.0
val_loss_count = 0
visualize_interval = 100
max_length = 3000
for idx in tqdm(range(min(len(dataset), max_length))):
    batch_data = dataset[idx]
    # add batch dimension
    imgs = batch_data["img"].unsqueeze(0).to(device)  # (1,3,h,w)
    cond_imgs = batch_data["cond_imgs"].unsqueeze(0).to(device)
    cond_masks = batch_data["cond_masks"].unsqueeze(0).to(device)
    list_targets = [_to_device(batch_data["targets"], device)]

    with torch.no_grad():
        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        input_imgs_size = [img.shape[-2:] for img in imgs]
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs, cond_imgs, cond_masks)
        val_losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            val_losses = model.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=list_targets,
            )
            block_postfix = model.block_postfix(i)
            val_losses = {f"{key}{block_postfix}": value for key, value in val_losses.items()}
            val_losses_all_blocks |= val_losses

        total_val_loss = model.criterion.loss_total(
            val_losses_all_blocks,
            log_fn=lambda *_args, **_kwargs: None,
            prefix="val",
        )
        val_loss_sum += float(total_val_loss.detach().cpu().item())
        val_loss_count += 1

        # Visualization every N steps 
        if idx % visualize_interval == 0:
            img_sizes = [img.shape[-2:] for img in imgs]
            # use the last block for visualization and only visualize the first batch item
            # scale up to original image size
            vis_mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
            vis_mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
                vis_mask_logits, img_sizes
            )
            for b in range(len(imgs)):
                pred = model.to_per_pixel_preds_instance(
                    vis_mask_logits[b],
                    class_logits_per_layer[-1][b],
                    model.mask_thresh,
                    model.overlap_thresh,
                )
                semantic_mask, instance_mask = pred[..., 0], pred[..., 1]

                pred_vis = visualize_image_mask_and_templates(
                    img=imgs[b], 
                    mask=semantic_mask,
                    cond_imgs=cond_imgs[b]
                )
                gt_vis = visualize_targets_and_templates(
                    img=imgs[b], 
                    targets=list_targets[b],
                    cond_imgs=cond_imgs[b]
                )
                break
            
            save_path = os.path.join(f"output/{dataset_name}_val_vis", f"{idx:06d}_pred.png")
            # concatenate gt and pred visualization
            concat_vis = np.concatenate([gt_vis, pred_vis], axis=1)
            os.makedirs(f"output/{dataset_name}_val_vis", exist_ok=True)
            plt.imsave(save_path, concat_vis.astype(np.uint8))

avg_total_val_loss = val_loss_sum / max(val_loss_count, 1)
print(f"avg_total_val_loss: {avg_total_val_loss:.6f}  (n={val_loss_count})")

       
