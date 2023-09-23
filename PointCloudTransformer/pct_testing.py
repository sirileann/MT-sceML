"""make sure to import all the classes you want to enter through the yaml file"""
import logging
import torch
from pytorch_lightning.cli import LightningCLI

from src.pct_multimodal import PCT_Cls


def cli_main():
    #  we pass the LightningModule by config file
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    # logging.basicConfig(filename='example.log', level=logging.ERROR)
    torch.set_default_dtype(torch.float32)
    cli_main()
