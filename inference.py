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



def get_observations(stats):
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
    pass

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data





def run_inference(obs_dict, networks, noise_scheduler, config, device):
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



if __name__ == '__main__':

    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats = np.load("stats.npy", allow_pickle=True).item()

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


    while True:
        
        obs_dict = get_observations(stats)

        action = run_inference(obs_dict, networks, config, device)

        '''
        TODO: Make the robot perform the action
        '''

        if (): # checks if the task is completed
            break






    