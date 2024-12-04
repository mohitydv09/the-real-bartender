import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import cv2


# transform the dataset

# class TransformData:
#     def __init__(self, original_data_folder):
#         # Create the transformed dataset directory
#         self.transformed_data_folder = 'transformed_dataset'
#         os.makedirs(self.transformed_data_folder, exist_ok=True)

#         # Read in the original dataset
#         self.original_data_folder = original_data_folder
#         npy_file_paths = sorted(glob.glob(os.path.join(original_data_folder, "transforms_*.npy")))
#         self.episodes = [np.load(path, allow_pickle=True).item() for path in npy_file_paths]

#         # Calculate episode lengths and the cumulative episode ends
#         self.episode_lengths = [len(ep) for ep in self.episodes]
#         self.episode_ends = np.cumsum(self.episode_lengths) - 1

#         # Create directories for each topic (camera images, states, and actions)
#         self.topics = [
#             'camera_thunder_wrist', 'camera_lightning_wrist', 'camera_both_front',
#             'lightning_gripper', 'lightning_angle', 'thunder_gripper', 'thunder_angle',
#             'spark_lightning_angle', 'spark_thunder_angle'
#         ]
#         for topic in self.topics:
#             os.makedirs(os.path.join(self.transformed_data_folder, topic), exist_ok=True)

#         # Process and store data
#         self._transform_and_store_data()

#         # Save episode ends as a numpy file
#         np.save(os.path.join(self.transformed_data_folder, 'episode_ends.npy'), self.episode_ends)


#     def _transform_and_store_data(self):
#         # Initialize a counter for naming the files
#         frame_counter = 0

#         # Process each episode and its frames
#         for episode in self.episodes:
#             for frame_idx, frame_data in episode.items():
#                 # Save the data for each topic as individual .npy files
#                 # Process and save camera images
#                 self._save_data('camera_thunder_wrist', frame_data['camera_thunder_wrist'], frame_counter)
#                 self._save_data('camera_lightning_wrist', frame_data['camera_lightning_wrist'], frame_counter)
#                 self._save_data('camera_both_front', frame_data['camera_both_front'], frame_counter)

#                 # Process and save states
#                 self._save_data('lightning_gripper', frame_data['lightning_gripper'], frame_counter)
#                 self._save_data('lightning_angle', frame_data['lightning_angle'], frame_counter)
#                 self._save_data('thunder_gripper', frame_data['thunder_gripper'], frame_counter)
#                 self._save_data('thunder_angle', frame_data['thunder_angle'], frame_counter)

#                 # Process and save actions
#                 self._save_data('spark_lightning_angle', frame_data['spark_lightning_angle'], frame_counter)
#                 self._save_data('spark_thunder_angle', frame_data['spark_thunder_angle'], frame_counter)

#                 # Increment the frame counter
#                 frame_counter += 1

#     def _save_data(self, topic, data, frame_counter):
#         """Helper function to save data into the respective folder"""
#         topic_folder = os.path.join(self.transformed_data_folder, topic)
#         np.save(os.path.join(topic_folder, f'{frame_counter}.npy'), data)

# Example usage:
# original_data_folder = '/home/rpmdt05/Code/the-real-bartender/dataset'  # Replace with your original dataset path
# transformer = TransformData(original_data_folder)
# print(transformer.episode_ends)


# transformed_data_folder = 'transformed_dataset'  # Path to the transformed dataset folder

# image_folder = os.path.join(transformed_data_folder, 'camera_both_front')
# image_files = sorted(
#     glob.glob(os.path.join(image_folder, "*.npy")),
#     key=lambda x: int(os.path.splitext(os.path.basename(x))[0])  # Extract the numeric part
# )
# print(image_files)
# images = [np.load(file) for file in image_files]

# for image in images:
#     cv2.imshow('image', image)
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()




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
                np.load(os.path.join('transformed_dataset', 'camera_both_front', f'{index}.npy')).transpose(2, 0, 1)
            )
            camera_lightning_wrist.append(
                np.load(os.path.join('transformed_dataset', 'camera_lightning_wrist', f'{index}.npy')).transpose(2, 0, 1)
            )
            camera_thunder_wrist.append(
                np.load(os.path.join('transformed_dataset', 'camera_thunder_wrist', f'{index}.npy')).transpose(2, 0, 1)
            )

            lightning_gripper.append(np.load(os.path.join('transformed_dataset', 'lightning_gripper', f'{index}.npy')))
            lightning_angle.append(np.load(os.path.join('transformed_dataset', 'lightning_angle', f'{index}.npy')))

            lightning_angle

            thunder_gripper.append(np.load(os.path.join('transformed_dataset', 'thunder_gripper', f'{index}.npy')))
            thunder_angle.append(np.load(os.path.join('transformed_dataset', 'thunder_angle', f'{index}.npy')))

        for index in pred_indices:
            spark_lightning_angle.append(np.load(os.path.join('transformed_dataset', 'spark_lightning_angle', f'{index}.npy')))
            spark_thunder_angle.append(np.load(os.path.join('transformed_dataset', 'spark_thunder_angle', f'{index}.npy')))

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


dataset_folder = '/home/rpmdt05/Code/the-real-bartender/transformed_dataset'  # Replace with your actual folder path
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
batch_size = 32  # Define the batch size
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











