"""
This script experiments with a custom dataloader to replicate the functionality demonstrated in the diffusion policy tutorial notebooks.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import cv2


class BartenderDataset(Dataset):
    def __init__(self, dataset_folder, obs_horizon, pred_horizon, action_horizon):
        
        self.episode_ends = np.load(os.path.join(dataset_folder, 'episode_ends.npy'))
        self.episode_starts = [0] + [end + 1 for end in self.episode_ends[:-1]]
        self.episode_ranges = list(zip(self.episode_starts, self.episode_ends))

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon

        self.sample_indices = torch.arange(0, self.episode_ends[-1] + 1)


    def __len__(self):
        return len(self.sample_indices)
    
    
    def __getitem__(self, idx):

        # identify which episode the index belongs to
        episode = np.where((np.array(self.episode_starts) <= idx) & (idx <= np.array(self.episode_ends)))[0]
        if len(episode) == 0:
            raise ValueError(f"Index {idx} is out of range of provided episodes.")
            
        episode_start = self.episode_starts[episode[0]]
        episode_end = self.episode_ends[episode[0]]

        '''
        Compute the indices for Observation Horizon
        '''
        obs_start = max(idx - self.obs_horizon + 1, episode_start)
        obs_indices = list(range(obs_start, idx + 1))
        
        # Pad with the episode start if needed
        while len(obs_indices) < self.obs_horizon:
            obs_indices.insert(0, episode_start)

        '''
        Compute the indices for Prediction Horizon
        '''
        pred_end = min(idx + self.pred_horizon-1, episode_end)
        pred_indices = list(range(idx, pred_end + 1))
        
        # Pad with the episode end if needed for prediction horizon
        while len(pred_indices) < self.pred_horizon:
            pred_indices.append(episode_end)


        '''
        Create the Data Item
        '''
        camera_both_front = []
        camera_lightning_wrist = []
        camera_thunder_wrist = []

        lightning_gripper = []
        lightning_angle = []

        thunder_gripper = []
        thunder_angle = []

        spark_lightning_angle = []
        spark_thunder_angle = []

        for index in obs_indices:
            camera_both_front.append(
                np.load(os.path.join('/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform', 'camera_both_front', f'{index}.npy')).transpose(2, 0, 1)
            )
            camera_lightning_wrist.append(
                np.load(os.path.join('/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform', 'camera_lightning_wrist', f'{index}.npy')).transpose(2, 0, 1)
            )
            camera_thunder_wrist.append(
                np.load(os.path.join('/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform', 'camera_thunder_wrist', f'{index}.npy')).transpose(2, 0, 1)
            )

            lightning_gripper.append(np.load(os.path.join('/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform', 'lightning_gripper', f'{index}.npy')))
            lightning_angle.append(np.load(os.path.join('/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform', 'lightning_angle', f'{index}.npy')))

            lightning_angle

            thunder_gripper.append(np.load(os.path.join('/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform', 'thunder_gripper', f'{index}.npy')))
            thunder_angle.append(np.load(os.path.join('/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform', 'thunder_angle', f'{index}.npy')))

        for index in pred_indices:
            spark_lightning_angle.append(np.load(os.path.join('/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform', 'spark_lightning_angle', f'{index}.npy')))
            spark_thunder_angle.append(np.load(os.path.join('/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform', 'spark_thunder_angle', f'{index}.npy')))

        return {
            'camera_both_front': np.array(camera_both_front),
            'camera_lightning_wrist': np.array(camera_lightning_wrist),
            'camera_thunder_wrist': np.array(camera_thunder_wrist),

            'lightning_gripper': np.array(lightning_gripper),
            'lightning_angle': np.array(lightning_angle),

            'thunder_gripper': np.array(thunder_gripper),
            'thunder_angle': np.array(thunder_angle),

            'spark_lightning_angle': np.array(spark_lightning_angle),
            'spark_thunder_angle': np.array(spark_thunder_angle)
        }


dataset_folder = '/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform'  # Replace with your actual folder path
obs_horizon = 4  # Define the observation horizon (how many time steps to observe)
pred_horizon = 16  # Define the prediction horizon (how many time steps to predict)
action_horizon = 8  # Define the action horizon (this can be used if needed for future predictions)

# Instantiate the BartenderDataset
dataset = BartenderDataset(
    dataset_folder=dataset_folder,
    obs_horizon=obs_horizon,
    pred_horizon=pred_horizon,
    action_horizon=action_horizon
)

# Create a DataLoader for batching the dataset
batch_size = 8  # Define the batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Get and print one batch from the DataLoader
batch_data = next(iter(dataloader))

print("One batch:")
print("camera_both_front shape:", batch_data['camera_both_front'].shape)
print("camera_lightning_wrist shape:", batch_data['camera_lightning_wrist'].shape)
print("camera_thunder_wrist shape:", batch_data['camera_thunder_wrist'].shape)
print("lightning_gripper shape:", batch_data['lightning_gripper'].shape)
print("lightning_angle shape:", batch_data['lightning_angle'].shape)
print("thunder_gripper shape:", batch_data['thunder_gripper'].shape)
print("thunder_angle shape:", batch_data['thunder_angle'].shape)
print("spark_lightning_angle shape:", batch_data['spark_lightning_angle'].shape)
print("spark_thunder_angle shape:", batch_data['spark_thunder_angle'].shape)


# images = batch_data['camera_both_front'][0]
# for image in images:
#     # print(image.shape)
#     image = np.transpose(image, (1, 2, 0))
#     image = np.asarray(image)
#     cv2.imshow('image', image)
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()











