from typing import Union
from torch.utils.data import DataLoader
from datasets.lightning_data_module import LightningDataModule
import yaml

from datasets.omni6d_dataset import Omni6DSegmentation
from datasets.ycbv_dataset import YCBVSegmentation

class OC_dataset(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (480, 640),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        # load dataset config
        config_path = "configs/dinov2/OC/config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.omni6d_config = self.config['Omni6dDataset']
        self.ycbv_config = self.config['YCBVDataset']
        self.augs = self.config['augs']

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        self.train_dataset = Omni6DSegmentation(
            Omni6dDataset=self.omni6d_config,
            augs=self.augs
        )
        self.val_dataset = YCBVSegmentation(YCBVDataset=self.ycbv_config)

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

