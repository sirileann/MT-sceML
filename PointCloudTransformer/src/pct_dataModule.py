import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

import numpy as np
import pandas as pd
import re
import json
from pathlib import Path
from PIL import Image


class BackDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.pcds = self.df['DIRS_PCDS'].tolist()
        self.labels = self.df['DIRS_LABELS'].tolist()

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):
        # Load image
        pcd_path = self.pcds[idx]
        pcd = np.load(pcd_path)
        pcd = torch.squeeze(torch.tensor(pcd, dtype=torch.float32))

        # Load target
        target_path = self.labels[idx]
        target = np.load(target_path)
        target = torch.squeeze(torch.tensor(target, dtype=torch.float32))

        # get ID
        path_id = re.search(r'backscan_pp_(\d+(?:_\d+)?)\.npy', str(self.pcds[idx])).group(1)

        # assert pcd.shape == torch.Size([1024, 3]), f"Image shape expected torch.Size([1024, 3]), got: {pcd.shape}"
        assert target.shape == torch.Size([60]), f"Target shape expected torch.Size([60]), got: {target.shape}"

        return pcd, target, path_id


class Back_DataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir="/home/siri/PycharmProjects/msc-sirirueegg-sceML/adapted_ResNet/data/test_dataset",
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
        back = sorted(Path(self.root_dir).glob('**/backscan_pp_*.npy'))
        labels = sorted(Path(self.root_dir).glob('**/isl_30_*.npy'))
        df = pd.DataFrame(list(zip(back, labels)), columns=['DIRS_PCDS', 'DIRS_LABELS'])

        assert self.train_per + self.test_per <= 1, "Invalid split percentages"

        df = df.sample(frac=1, random_state=pl.seed_everything()).reset_index(drop=True)

        nr_train = int(len(df) * self.train_per)
        nr_test = int(len(df) * self.test_per)
        nr_val = len(df) - nr_train - nr_test

        if stage == "fit":
            self.train_ds = BackDataset(df=df[:nr_train])
            self.val_ds = BackDataset(df=df[nr_train + nr_test:])
            assert len(self.val_ds) == nr_val
        if stage == "test":
            self.test_ds = BackDataset(df=df[nr_train: nr_train + nr_test:])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, num_workers=self.numWorkers, batch_size=self.batch_size,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, num_workers=self.numWorkers, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, num_workers=self.numWorkers, batch_size=1)


class FixPointsDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.pcds = self.df['DIRS_PCDS'].tolist()
        self.labels = self.df['DIRS_LABELS'].tolist()

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):
        # Load fixPoints
        with open(self.pcds[idx], "r") as f:
            fixPoints_dict = json.load(f)
        pcd = np.array(list(fixPoints_dict.values()))
        pcd = torch.squeeze(torch.tensor(pcd, dtype=torch.float32))

        # Load target
        target_path = self.labels[idx]
        target = np.load(target_path)
        target = torch.squeeze(torch.tensor(target, dtype=torch.float32))

        # get ID
        path_id = re.search(r'fixPoints_pp_(\d+(?:_\d+)?)\.json', str(self.pcds[idx])).group(1)

        assert pcd.shape == torch.Size([12, 3]), f"FixPoints shape expected torch.Size([12, 3]), got: {pcd.shape}"
        assert target.shape == torch.Size([60]), f"Target shape expected torch.Size([60]), got: {target.shape}"

        return pcd, target, path_id


class FixPoint_DataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir="/home/siri/PycharmProjects/msc-sirirueegg-sceML/PintCloudTransformer/data/test_dataset",
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
        depth_maps = sorted(Path(self.root_dir).glob('**/fixPoints_pp_*.json'))
        labels = sorted(Path(self.root_dir).glob('**/isl_30_*.npy'))
        df = pd.DataFrame(list(zip(depth_maps, labels)), columns=['DIRS_PCDS', 'DIRS_LABELS'])

        assert self.train_per + self.test_per <= 1, "Invalid split percentages"

        df = df.sample(frac=1, random_state=pl.seed_everything()).reset_index(drop=True)

        nr_train = int(len(df) * self.train_per)
        nr_test = int(len(df) * self.test_per)
        nr_val = len(df) - nr_train - nr_test

        if stage == "fit":
            self.train_ds = FixPointsDataset(df=df[:nr_train])
            self.val_ds = FixPointsDataset(df=df[nr_train + nr_test:])
            assert len(self.val_ds) == nr_val
        if stage == "test":
            self.test_ds = FixPointsDataset(df=df[nr_train: nr_train + nr_test:])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, num_workers=self.numWorkers, batch_size=self.batch_size,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, num_workers=self.numWorkers, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, num_workers=self.numWorkers, batch_size=1)


class BackPCBackDMDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.images = self.df['DIRS_IMAGES'].tolist()
        self.pcds = self.df['DIRS_PCDS'].tolist()
        self.labels = self.df['DIRS_LABELS'].tolist()

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):
        # Load pcd
        pcd_path = self.pcds[idx]
        pcd = np.load(pcd_path)
        pcd = torch.squeeze(torch.tensor(pcd, dtype=torch.float32))

        # Load image
        img_path = self.images[idx]
        img = Image.open(img_path)
        img = img.convert('L')
        img = ToTensor()(img)

        # Load target
        target_path = self.labels[idx]
        target = np.load(target_path)
        target = torch.squeeze(torch.tensor(target, dtype=torch.float32))

        # get ID
        path_id = re.search(r'backscan_pp_(\d+(?:_\d+)?)\.npy', str(self.pcds[idx])).group(1)

        assert pcd.shape == torch.Size([1024, 3]), f"Image shape expected torch.Size([1024, 3]), got: {pcd.shape}"
        assert target.shape == torch.Size([60]), f"Target shape expected torch.Size([60]), got: {target.shape}"

        return pcd, img, target, path_id


class BackPC_BackDM_DataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir="/home/siri/PycharmProjects/msc-sirirueegg-sceML/adapted_ResNet/data/test_dataset",
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
        pcds = sorted(Path(self.root_dir).glob('**/backscan_pp_*.npy'))
        depth_maps = sorted(Path(self.root_dir).glob('**/depthmap_*.jpeg'))
        labels = sorted(Path(self.root_dir).glob('**/isl_30_*.npy'))
        df = pd.DataFrame(list(zip(pcds, depth_maps, labels)), columns=['DIRS_PCDS', 'DIRS_IMAGES', 'DIRS_LABELS'])

        assert self.train_per + self.test_per <= 1, "Invalid split percentages"

        df = df.sample(frac=1, random_state=pl.seed_everything()).reset_index(drop=True)

        nr_train = int(len(df) * self.train_per)
        nr_test = int(len(df) * self.test_per)
        nr_val = len(df) - nr_train - nr_test

        if stage == "fit":
            self.train_ds = BackPCBackDMDataset(df=df[:nr_train])
            self.val_ds = BackPCBackDMDataset(df=df[nr_train + nr_test:])
            assert len(self.val_ds) == nr_val
        if stage == "test":
            self.test_ds = BackPCBackDMDataset(df=df[nr_train: nr_train + nr_test:])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, num_workers=self.numWorkers, batch_size=self.batch_size,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, num_workers=self.numWorkers, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, num_workers=self.numWorkers, batch_size=1)


class BackFixDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.pcds = self.df['DIRS_PCDS'].tolist()
        self.images = self.df['DIRS_POINTS'].tolist()
        self.labels = self.df['DIRS_LABELS'].tolist()

        # print(len(self.images), len(self.pcds), len(self.labels))

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):
        # Load pcd
        pcd_path = self.pcds[idx]
        pcd = np.load(pcd_path)
        pcd = torch.squeeze(torch.tensor(pcd, dtype=torch.float32))

        # Load fixPoints
        with open(self.images[idx], "r") as f:
            fixPoints_dict = json.load(f)
        pcd2 = np.array(list(fixPoints_dict.values()))
        pcd2 = torch.squeeze(torch.tensor(pcd2, dtype=torch.float32))

        # Load target
        target_path = self.labels[idx]
        target = np.load(target_path)
        target = torch.squeeze(torch.tensor(target, dtype=torch.float32))

        # get ID
        path_id = re.search(r'backscan_pp_(\d+(?:_\d+)?)\.npy', str(self.pcds[idx])).group(1)

        assert pcd.shape == torch.Size([1024, 3]), f"Image shape expected torch.Size([1024, 3]), got: {pcd.shape}"
        assert target.shape == torch.Size([60]), f"Target shape expected torch.Size([60]), got: {target.shape}"

        return pcd, pcd2, target, path_id


class Back_Fix_DataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir="/home/siri/PycharmProjects/msc-sirirueegg-sceML/adapted_ResNet/data/test_dataset",
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
        pcds = sorted(Path(self.root_dir).glob('**/backscan_pp_*.npy'))
        fixPoints = sorted(Path(self.root_dir).glob('**/fixPoints_pp_*.json'))
        labels = sorted(Path(self.root_dir).glob('**/isl_30_*.npy'))
        df = pd.DataFrame(list(zip(pcds, fixPoints, labels)), columns=['DIRS_PCDS', 'DIRS_POINTS', 'DIRS_LABELS'])

        assert self.train_per + self.test_per <= 1, "Invalid split percentages"

        df = df.sample(frac=1, random_state=pl.seed_everything()).reset_index(drop=True)

        nr_train = int(len(df) * self.train_per)
        nr_test = int(len(df) * self.test_per)
        nr_val = len(df) - nr_train - nr_test

        if stage == "fit":
            self.train_ds = BackFixDataset(df=df[:nr_train])
            self.val_ds = BackFixDataset(df=df[nr_train + nr_test:])
            assert len(self.val_ds) == nr_val
        if stage == "test":
            self.test_ds = BackFixDataset(df=df[nr_train: nr_train + nr_test:])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, num_workers=self.numWorkers, batch_size=self.batch_size,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, num_workers=self.numWorkers, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, num_workers=self.numWorkers, batch_size=1)


class BackESLFixDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.backs = self.df['DIRS_BACK'].tolist()
        self.esls = self.df['DIRS_ESL'].tolist()
        self.fixs = self.df['DIRS_FIX'].tolist()
        self.labels = self.df['DIRS_LABELS'].tolist()

    def __len__(self):
        return len(self.backs)

    def __getitem__(self, idx):
        # Load pcd
        pcd_path = self.backs[idx]
        pcd = np.load(pcd_path)
        pcd = torch.squeeze(torch.tensor(pcd, dtype=torch.float32))

        # Load fixPoints
        pcd2_path = self.esls[idx]
        pcd2 = np.load(pcd2_path)
        pcd2 = torch.squeeze(torch.tensor(pcd2, dtype=torch.float32))

        # Load fixPoints
        with open(self.fixs[idx], "r") as f:
            fixPoints_dict = json.load(f)
        pcd3 = np.array(list(fixPoints_dict.values()))
        pcd3 = torch.squeeze(torch.tensor(pcd3, dtype=torch.float32))

        # Load target
        target_path = self.labels[idx]
        target = np.load(target_path)
        target = torch.squeeze(torch.tensor(target, dtype=torch.float32))

        # get ID
        path_id = re.search(r'backscan_pp_(\d+(?:_\d+)?)\.npy', str(self.backs[idx])).group(1)

        assert pcd.shape == torch.Size([1024, 3]), f"Image shape expected torch.Size([1024, 3]), got: {pcd.shape}"
        assert target.shape == torch.Size([60]), f"Target shape expected torch.Size([60]), got: {target.shape}"

        return pcd, pcd2, pcd3, target, path_id


class Back_ESL_Fix_DataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir="/home/siri/PycharmProjects/msc-sirirueegg-sceML/adapted_ResNet/data/test_dataset",
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
        back = sorted(Path(self.root_dir).glob('**/backscan_pp_*.npy'))
        esl = sorted(Path(self.root_dir).glob('**/esl_pp_*.npy'))
        fix = sorted(Path(self.root_dir).glob('**/fixPoints_pp_*.json'))
        labels = sorted(Path(self.root_dir).glob('**/isl_30_*.npy'))
        df = pd.DataFrame(list(zip(back, esl, fix, labels)),
                          columns=['DIRS_BACK', 'DIRS_ESL', 'DIRS_FIX', 'DIRS_LABELS'])

        assert self.train_per + self.test_per <= 1, "Invalid split percentages"

        df = df.sample(frac=1, random_state=pl.seed_everything()).reset_index(drop=True)

        nr_train = int(len(df) * self.train_per)
        nr_test = int(len(df) * self.test_per)
        nr_val = len(df) - nr_train - nr_test

        if stage == "fit":
            self.train_ds = BackESLFixDataset(df=df[:nr_train])
            self.val_ds = BackESLFixDataset(df=df[nr_train + nr_test:])
            assert len(self.val_ds) == nr_val
        if stage == "test":
            self.test_ds = BackESLFixDataset(df=df[nr_train: nr_train + nr_test:])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, num_workers=self.numWorkers, batch_size=self.batch_size,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, num_workers=self.numWorkers, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, num_workers=self.numWorkers, batch_size=1)
