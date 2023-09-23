import logging
import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Trainer

from src.ConvNets import ConvNets
from src.ConvNets_dataModule import ConvNetDataModule


def cli_main():
    #  we pass the LightningModule by config file
    LightningCLI(datamodule_class=ConvNetDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    # logging.basicConfig(filename='example.log', level=logging.ERROR)
    torch.set_default_dtype(torch.float32)
    cli_main()
