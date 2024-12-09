import os
import wandb
import torch
import yaml
import numpy as np
from tqdm import tqdm
import numpy as np
import time


from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scripts.dataset import BartenderDataset
from scripts.network import ConditionalUnet1D
from scripts.vision_encoder import get_resnet, replace_bn_with_gn
from scripts.get_observations import ObservationSubscriber

import rospy
from std_msgs.msg import Float32MultiArray 
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from collections import deque

import config_.yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_network(config, device):
    '''
    Load the Models
    '''
    vision_encoder_front = replace_bn_with_gn(get_resnet('resnet18'))
    vision_encoder_thunder_wrist = replace_bn_with_gn(get_resnet('resnet18'))
    vision_encoder_lightning_wrist = replace_bn_with_gn(get_resnet('resnet18'))

    vision_feature_dim = 512 * 3
    state_dim = 14
    observation_dim = vision_feature_dim + state_dim
    action_dim = 14

    noise_prediction_network = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=observation_dim * config['observation_horizon'],
    )

    networks = torch.nn.ModuleDict({
        'vision_encoder_front': vision_encoder_front,
        'vision_encoder_thunder_wrist': vision_encoder_thunder_wrist,
        'vision_encoder_lightning_wrist': vision_encoder_lightning_wrist,
        'noise_prediction_network': noise_prediction_network
    }).to(device)

    checkpoint_path = "models/uncork_v2.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    networks.load_state_dict(state_dict)

    return networks


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


def get_observations(observation_object, config, stats):
    '''
    TODO: Stream throuch the intel realsense camera, subscribe to the ros topics
    and get the observations.
    Think about storing a queue and when called, return the last n observations, based on the horizon.
    
    Output:
            Images: (obs_horizon, C, H, W) -> normalized
            Agent State: (obs_horizon, 14) -> normalized

    format:
        {
            'img_front': img_front,
            'img_wrist_thunder': img_wrist_thunder,
            'img_wrist_lightning': img_wrist_lightning,
            'agent_pos': agent_pos,
        }
    '''
    
    obs_dict = observation_object.get_last_n_observations()

    img_front = obs_dict['Images']['img_front']
    img_wrist_thunder = obs_dict['Images']['img_wrist_thunder']
    img_wrist_lightning = obs_dict['Images']['img_wrist_lightning']
    agent_pos = obs_dict['agent_state']

    # Normalize the images
    img_front = img_front.astype(np.float32) / 255.0
    img_wrist_thunder = img_wrist_thunder.astype(np.float32) / 255.0
    img_wrist_lightning = img_wrist_lightning.astype(np.float32) / 255.0
    # change image to torch and change the axis to (C, H, W)
    img_front = torch.from_numpy(img_front).permute(0, 3, 1, 2)
    img_wrist_thunder = torch.from_numpy(img_wrist_thunder).permute(0, 3, 1, 2)
    img_wrist_lightning = torch.from_numpy(img_wrist_lightning).permute(0, 3, 1, 2)

    # Normalize the agent state
    agent_pos = normalize_data(agent_pos, stats = stats['agent_state'])



def run_inference(obs_dict, networks, noise_scheduler, stats, config, device):
    B = 1
    pred_horizon = config['prediction_horizon']
    action_dim = config['action_dim']
    nimage_front = obs_dict['img_front'].to(device) ## (obs_horizon, C, H, W)
    nimage_thunder_wrist = obs_dict['img_wrist_thunder'].to(device) ## (obs_horizon, C, H, W)
    nimage_lightning_wrist = obs_dict['img_wrist_lightning'].to(device) ## (obs_horizon, C, H, W)

    nagent_state = obs_dict['agent_pos'].to(device) ## (obs_horizon, 14)

    with torch.no_grad():
        
        # Get the global condition
        img_front_features = networks['vision_encoder_front'](nimage_front) ## (obs_horizon, D)
        img_thunder_wrist_features = networks['vision_encoder_thunder_wrist'](nimage_thunder_wrist) ## (obs_horizon, D)
        img_lightning_wrist_features = networks['vision_encoder_lightning_wrist'](nimage_lightning_wrist) ## (obs_horizon, D)


        obs_features = torch.cat([img_front_features, img_thunder_wrist_features, img_lightning_wrist_features, nagent_state], dim=-1) ## (obs_horizon, 512 * 3 + 14)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        # Sample the noisy action
        noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
        naction = noisy_action

        noise_scheduler.set_timesteps(config['num_diffusion_iters'])

        for k in noise_scheduler.timesteps:
            noise_pred = networks['noise_prediction_network'](
                naction, 
                k, 
                global_cond=obs_cond
            )

            naction = noise_scheduler.step(
                model_output = noise_pred,
                timestep = k,
                sample = naction
            ).prev_sample

    naction = naction.detach().to('cpu').numpy()# (B, pred_horizon, action_dim)
    naction = naction[0] ## (pred_horizon, action_dim)
    naction = unnormalize_data(naction, stats = stats['action'])

    # take action horizon number of actions
    start = config['observation_horizon'] - 1
    end = start + config['action_horizon']
    action = naction[start:end, :]

    return action



def main():

    config = load_config('config.yaml')
    device = config['device']
    stats = np.load("stats.npy", allow_pickle=True).item()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config['num_diffusion_iters'],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    networks = load_network(config, device)

    observation_subscriber = ObservationSubscriber(obs_horizon=config['observation_horizon'])
    rospy.spin()
    time.sleep(2)


    while True:
        
        obs_dict = get_observations(observation_subscriber, config, stats)
        action = run_inference(obs_dict, networks, noise_scheduler, stats, config, device)

        '''
        TODO: Make the robot perform the action
        '''

        if (): # checks if the task is completed
            break


if __name__ == '__main__':
    main()