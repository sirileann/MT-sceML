"""make sure to import all the classes you want to enter through the yaml file"""
import logging
import torch
from pytorch_lightning.cli import LightningCLI

from src.pct_multimodal import PCT_Cls


def cli_main():
    #  we pass the LightningModule by config file
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
    trainer = cli.trainer
    trainer.test(datamodule=cli.datamodule, ckpt_path="best")  # Run test after fitting


if __name__ == "__main__":
    # logging.basicConfig(filename='example.log', level=logging.ERROR)
    torch.set_default_dtype(torch.float32)
    cli_main()
