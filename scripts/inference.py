import os
import wandb
import torch
import yaml
import numpy as np
from tqdm import tqdm

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scripts.dataset import BartenderDataset
from scripts.network import ConditionalUnet1D
from scripts.vision_encoder import get_resnet, replace_bn_with_gn


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config("config_.yaml")


'''
Load the Models
'''
vision_encoder_front = replace_bn_with_gn(get_resnet('resnet18'))
vision_encoder_thunder_wrist = replace_bn_with_gn(get_resnet('resnet18'))
vision_encoder_lightning_wrist = replace_bn_with_gn(get_resnet('resnet18'))
