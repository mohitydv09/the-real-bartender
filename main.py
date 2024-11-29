import wandb
import torch
import yaml
import numpy as np
from tqdm import tqdm

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scripts.dataset import PushTImageDataset
from scripts.network import ConditionalUnet1D
from scripts.vision_encoder import get_resnet, replace_bn_with_gn

def forward_pass(batch, networks, noise_scheduler, device, obs_horizon):
    nimage = batch['image'][:,:obs_horizon].to(device) ## (B, obs_horizon, H, W, C)
    nagent_state = batch['agent_pos'][:,:obs_horizon].to(device) ## (B, obs_horizon, 2)
    naction = batch['action'].to(device) ## (B, action_horizon, 2)

    image_features = networks['vision_encoder'](nimage.flatten(end_dim=1)) ## (B * obs_horizon, D)
    image_features = image_features.reshape(*nimage.shape[:2], -1) ## (B, obs_horizon, D)

    obs_features = torch.cat([image_features, nagent_state], dim=-1) ## (B, obs_horizon, D + 2)
    obs_cond = obs_features.flatten(start_dim=1) ## (B, obs_horizon * (D + 2))

    ## Diffusion Iteration
    B = nagent_state.shape[0]
    noise = torch.randn(naction.shape, device=device)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
    noise_pred = networks['noise_prediction_network'](noisy_actions, timesteps, global_cond=obs_cond)

    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    return loss

def train_epoch(dataloader, 
                networks, 
                optimizer, 
                lr_scheduler, 
                ema, 
                noise_scheduler, 
                device, 
                obs_horizon):
    epoch_loss = []
    networks.train()

    with tqdm(dataloader, desc="Training") as tepoch:
        for batch in tepoch:
            loss  = forward_pass(batch, 
                                 networks,
                                 noise_scheduler,
                                 device,
                                 obs_horizon)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(networks.parameters())

            ## Logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            tepoch.set_postfix(loss=loss_cpu)
    
    return np.mean(epoch_loss)


def train(config):
    device = config['device']

    wandb.init(project=config['wandb_project'], config=config)
    config = wandb.config

    ## Create Dataset and Dataloader
    dateset = PushTImageDataset(
        dataset_path=config['dataset_path'],
        pred_horizon=config['prediction_horizon'],
        obs_horizon=config['observation_horizon'],
        action_horizon=config['action_horizon'],
    )
    stats = dateset.stats

    dataloader = torch.utils.data.DataLoader(
        dataset=dateset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,            ## Accelerate cpu-gpu transfer
        persistent_workers=True,    ## don't kill workers after each epoch
    )

    ## Model & Optimizer
    vision_encoder = replace_bn_with_gn(get_resnet('resnet18'))

    ## Output of resnet18 is 512
    vision_feature_dim = 512
    state_dim = 2
    observation_dim = vision_feature_dim + state_dim
    action_dim = 2

    noise_prediction_network = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=observation_dim * config['observation_horizon'],
    )

    networks = torch.nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_prediction_network': noise_prediction_network
    }).to(device)

    ## Optimizer
    optimizer = torch.optim.AdmaW(
        params=networks.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warump_steps=config['num_warmup_steps'],
        num_training_steps=len(dataloader) * config['epochs']
    )

    ema = EMAModel(parameters=networks.parameters(), power=0.75)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config['num_diffusion_iters'],
        bets_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    ## Training Loop
    for epoch in range(config['epochs']):
        epoch_loss = train_epoch(
            dataloader, 
            networks, 
            optimizer, 
            lr_scheduler, 
            ema, 
            noise_scheduler, 
            device, 
            config['observation_horizon']
        )
        wandb.log({'epoch': epoch, 'loss': epoch_loss})
        print(f"Epoch {epoch}/{config['epochs']}, Loss: {epoch_loss}")
    
    torch.save(networks.state_dict(), config['model_path'])
    wandb.save(config['model_path'])
    print("Model saved to ", config['model_path'])

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config("config.yaml")
    train(config)

if __name__ == '__main__':
    main()
