import os, sys, argparse, pdb
sys.path.append(os.getcwd())

from importlib import import_module
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from typing import List
from copy import deepcopy
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torchvision import transforms as T
from torch.utils.data import DataLoader

from core.kpe_model import KPEModel, ImageLogger
from core.loader import PoseDatasetPickle
from core.utils import get_obj_from_str, instantiate_from_config

def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Path to config file')
    parser.add_argument("--disable_logger", required=False, action='store_true')
    parser.add_argument("--finetune_from", type=str, default=None)
    parser.add_argument("-t", "--train", type=str2bool, default=False)

    return parser


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args, unknown_args = parser.parse_known_args()
    config = OmegaConf.load(args.config)

    # logger
    project_name = os.path.basename(args.config).split('.')[0]
    time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = f"logs/{project_name}/{time_now}/"
    os.makedirs(log_dir, exist_ok=True)
    if not args.disable_logger:
        logger = WandbLogger(project=project_name, log_model=True,
                                save_dir=log_dir)
    else:
        logger = None

    ckpt_cb = ModelCheckpoint(dirpath=log_dir, 
                        monitor='val/loss_image', 
                        save_top_k=1,
                        save_last=True)
    

    # Dataloader
    train_dataset = get_obj_from_str(config.train_dataset.target)(\
        text_encoder_config=config.text_encoder,
        pose_encoder_config=config.pose_encoder,
        **config.train_dataset.params)

    train_loader = DataLoader(train_dataset, shuffle=True, **config.train_loader)
    
    val_dataset = get_obj_from_str(config.val_dataset.target)(\
        text_encoder_config=config.text_encoder,
        pose_encoder_config=config.pose_encoder,
        **config.val_dataset.params)

    val_loader = DataLoader(val_dataset, shuffle=False, **config.val_loader)
    
    test_dataset = get_obj_from_str(config.test_dataset.target)(\
        text_encoder_config=config.text_encoder,
        pose_encoder_config=config.pose_encoder,
        **config.test_dataset.params)

    test_loader = DataLoader(test_dataset, shuffle=False, **config.val_loader)
        

    # Trainer
    lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')
    image_log_cb = ImageLogger(config.logging.image_frequency, config.logging.max_images)

    trainer_config = config.get("trainer", OmegaConf.create())
    trainer = Trainer.from_argparse_args(args, 
                    logger=logger,
                    callbacks=[ckpt_cb, lr_monitor_cb, image_log_cb],
                    strategy=DDPStrategy(find_unused_parameters=False),
                    **trainer_config)

    # Model
    config.optimizer.params.lr*=trainer.num_devices
    print(f"Resetting learning rate to {config.optimizer.params.lr:.6f}")
    model = KPEModel(config)

    if not args.finetune_from == "":
        print(f"Attempting to load state from {args.finetune_from}")
        old_state = torch.load(args.finetune_from, map_location="cpu")
        if "state_dict" in old_state:
            print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
            old_state = old_state["state_dict"]
        m, u = model.load_state_dict(old_state, strict=False)
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)

    if args.train:
        trainer.fit(model, train_loader, val_loader)

    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    main()