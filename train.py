import argparse
import collections
import warnings

import numpy as np
import torch

import hw_nv.loss as module_loss
import hw_nv.model as module_arch
from hw_nv.trainer import Trainer
from hw_nv.utils import prepare_device
from hw_nv.utils.object_loading import get_dataloaders
from hw_nv.utils.parse_config import ConfigParser


import os

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    # model = config.init_obj(config["arch"], module_arch) #TODO clear old code
    generator = config.init_obj(config["arch"]["generator"], module_arch)
    discriminator = config.init_obj(config["arch"]["discriminator"], module_arch)
    # logger.info(generator)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    # model = model.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    if len(device_ids) > 1:
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

    # get function handles of loss
    loss_module_gen = config.init_obj(config["loss_gen"], module_loss).to(device)
    loss_module_disc = config.init_obj(config["loss_disc"], module_loss).to(device)


    #creating dir for generated audio samples
    os.makedirs("results", exist_ok=True)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params_gen = filter(lambda p: p.requires_grad, generator.parameters())
    optimizer_gen = config.init_obj(config["optimizer_gen"], torch.optim, trainable_params_gen)
    trainable_params_disc = filter(lambda p: p.requires_grad, discriminator.parameters())
    optimizer_disc = config.init_obj(config["optimizer_disc"], torch.optim, trainable_params_disc)

    lr_scheduler_gen = config.init_obj(config["lr_scheduler_gen"], torch.optim.lr_scheduler, optimizer_gen)
    lr_scheduler_disc = config.init_obj(config["lr_scheduler_disc"], torch.optim.lr_scheduler, optimizer_disc)

    trainer = Trainer(
        generator,
        discriminator,
        loss_module_gen,
        loss_module_disc,
        optimizer_gen,
        optimizer_disc,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler_gen=lr_scheduler_gen,
        lr_scheduler_disc=lr_scheduler_disc,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-l",
        "--load",
        default=None,
        type=str,
        help="path to checkpoint (default: None)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
