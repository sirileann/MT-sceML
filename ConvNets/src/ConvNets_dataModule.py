"""
@Author: Siri Leane Rüegg
@Contact: sirrueeg@ethz.ch
@File: ConvNets_dataModule.py
"""

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from pathlib import Path
import re
import pandas as pd
import numpy as np
from PIL import Image


class ConvNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir="/home/siri/PycharmProjects/msc-sirirueegg-sceML/ConvNets/data/test_dataset",
                 train_per=0.8,
                 test_per=0.1,
                 batch_size: int = 8,
                 numWorkers: int = 4):
        super().__init__()
        self.root_dir = root_dir
        self.train_per = train_per
        self.test_per = test_per
        self.batch_size = batch_size
        self.numWorkers = numWorkers

    def setup(self, stage: str):
        depth_maps = sorted(Path(self.root_dir).glob('**/depthmap_*.jpeg'))
        labels = sorted(Path(self.root_dir).glob('**/isl_30_*.npy'))
        df = pd.DataFrame(list(zip(depth_maps, labels)), columns=['DIRS_IMAGES', 'DIRS_LABELS'])

        assert self.train_per + self.test_per <= 1.0, "Invalid split percentages"

        df = df.sample(frac=1, random_state=pl.seed_everything()).reset_index(drop=True)

        nr_train = int(len(df) * self.train_per)
        nr_test = int(len(df) * self.test_per)
        nr_val = len(df) - nr_train - nr_test

        if stage == "fit":
            self.train_ds = CroatianDataset(df=df[:nr_train])
            self.val_ds = CroatianDataset(df=df[nr_train + nr_test:])
            assert len(self.val_ds) == nr_val
        if stage == "test":
            self.test_ds = CroatianDataset(df=df[nr_train: nr_train + nr_test:])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, num_workers=self.numWorkers, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, num_workers=self.numWorkers, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, num_workers=self.numWorkers, batch_size=1)


class CroatianDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.depth_maps = self.df['DIRS_IMAGES'].tolist()
        self.labels = self.df['DIRS_LABELS'].tolist()

    def __len__(self):
        return len(self.depth_maps)

    def __getitem__(self, idx):
        # Load image
        img_path = self.depth_maps[idx]
        img = Image.open(img_path)
        img = img.convert('L')
        img = ToTensor()(img)

        # Load target
        target_path = self.labels[idx]
        target = np.load(target_path)
        target = torch.squeeze(torch.tensor(target, dtype=torch.float32))

        # get ID
        path_id = re.search(r'depthmap_(\d+(?:_\d+)?)\.jpeg', str(self.depth_maps[idx])).group(1)

        assert img.shape == torch.Size(
            [1, 640, 480]), f"Image shape expected torch.Size([1, 640, 480]), got: {img.shape}"
        assert target.shape == torch.Size([60]), f"Target shape expected torch.Size([60]), got: {target.shape}"

        return img, target, path_id
