import argparse
import collections
import warnings

import numpy as np
import torch

import hw_nv.loss as module_loss
import hw_nv.data as module_data
import hw_nv.metric as module_metric
import hw_nv.model as module_arch
from hw_nv.trainer import Trainer
from hw_nv.utils import prepare_device
from hw_nv.utils.parse_config import ConfigParser

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

    dataset = config.init_obj(config['dataset'], module_data)
    train_data_loader = config.init_obj(
        config['dataloader'],
        torch.utils.data,
        dataset=dataset,
        collate_fn=dataset.collate
    )

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss = config.init_obj(config["loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler

    def get_params(model):
        return list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizers = {
        'optimizer_g': config.init_obj(
            config["optimizer_g"],
            torch.optim,
            get_params(model.generator)),
        'optimizer_d': config.init_obj(
            config["optimizer_d"],
            torch.optim,
            get_params(model.mp_discriminator) + get_params(model.ms_discriminator)),
    }

    lr_schedulers = dict()
    for name, opt in optimizers.items():
        lr_schedulers[name] = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, opt)

    trainer = Trainer(
        model,
        loss,
        optimizers,
        config=config,
        device=device,
        dataloader=train_data_loader,
        lr_schedulers=lr_schedulers,
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