import os
import zarr
import numpy as np
import glob
from tqdm import tqdm

class TransformData:
    def __init__(self, original_data_folder):
        # Create the transformed dataset directory
        self.transformed_data_folder = 'transformed_data/initial_transform'
        os.makedirs(self.transformed_data_folder, exist_ok=True)

        # Read in the original dataset folder
        self.original_data_folder = original_data_folder
        self.npy_file_paths = sorted(
            glob.glob(os.path.join(original_data_folder, "transforms_*.npy")), 
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
        )

        # Initialize episode ends list
        self.episode_ends = []

        # Create directories for each topic
        self.topics = [
            'camera_thunder_wrist', 'camera_lightning_wrist', 'camera_both_front',
            'lightning_gripper', 'lightning_angle', 'thunder_gripper', 'thunder_angle',
            'spark_lightning_angle', 'spark_thunder_angle'
        ]
        for topic in self.topics:
            os.makedirs(os.path.join(self.transformed_data_folder, topic), exist_ok=True)

        # Process and store data incrementally
        self._transform_and_store_data()

        # Save episode ends as a numpy file
        np.save(os.path.join(self.transformed_data_folder, 'episode_ends.npy'), np.array(self.episode_ends))


    def _transform_and_store_data(self):
        frame_counter = 0

        # Process each file one by one
        for file_idx, npy_file in tqdm(enumerate(self.npy_file_paths)):
            # Load the current .npy file
            episode = np.load(npy_file, allow_pickle=True).item()

            # Process each frame in the episode
            len_episode = len(episode.items())
            for frame_idx, frame_data in episode.items():
                if (frame_idx < 20 or frame_idx > len_episode - 20):
                    ## Skip the first 20 and last 20 frames
                    continue
                # Save the data for each topic as individual .npy files
                self._save_data('camera_thunder_wrist', frame_data['camera_thunder_wrist'], frame_counter)
                self._save_data('camera_lightning_wrist', frame_data['camera_lightning_wrist'], frame_counter)
                self._save_data('camera_both_front', frame_data['camera_both_front'], frame_counter)

                self._save_data('lightning_gripper', frame_data['lightning_gripper'], frame_counter)
                self._save_data('lightning_angle', frame_data['lightning_angle'], frame_counter)
                self._save_data('thunder_gripper', frame_data['thunder_gripper'], frame_counter)
                self._save_data('thunder_angle', frame_data['thunder_angle'], frame_counter)

                self._save_data('spark_lightning_angle', frame_data['spark_lightning_angle'], frame_counter)
                self._save_data('spark_thunder_angle', frame_data['spark_thunder_angle'], frame_counter)

                # Increment the frame counter
                frame_counter += 1

            # Track the end of this episode
            self.episode_ends.append(frame_counter - 1)

    def _save_data(self, topic, data, frame_counter):
        """Helper function to save data into the respective folder"""
        topic_folder = os.path.join(self.transformed_data_folder, topic)
        np.save(os.path.join(topic_folder, f'{frame_counter}.npy'), data)

def data2zarr():
    ## Read the data from the directory
    raw_data_directory = 'transformed_data/initial_transform'

    num_of_data_points = len(os.listdir(os.path.join(raw_data_directory, 'camera_thunder_wrist')))
    print("Number of data points: ", num_of_data_points)
    N = num_of_data_points
    image_shape = np.load("transformed_data/initial_transform/camera_both_front/0.npy").shape
    H, W, C = image_shape

    ## Create a new Zarr Group
    zarr_data_file_path = 'transformed_data/uncork_v2.zarr'
    root = zarr.open_group(zarr_data_file_path, mode='w')

    ## Create data group
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    ## Fill the data
    episode_ends_raw = np.load("transformed_data/initial_transform/episode_ends.npy")
    meta_group.create_dataset('episode_ends', shape=episode_ends_raw.shape, dtype='int64')
    meta_group['episode_ends'][:] = episode_ends_raw

    N = episode_ends_raw[-1] + 1

    data_group.create_dataset('img_front', shape=(N, H, W, C), chunks=(1, H, W, C), dtype='float32')
    data_group.create_dataset('img_wrist_thunder', shape=(N, H, W, C), chunks=(1, H, W, C), dtype='float32')
    data_group.create_dataset('img_wrist_lightning', shape=(N, H, W, C), chunks=(1, H, W, C), dtype='float32')
    data_group.create_dataset('states', shape=(N,14), dtype='float32')    ## UR angles
    data_group.create_dataset('actions', shape=(N,14), dtype='float32')   ## Spark Angles

    states = np.zeros((N, 14), dtype='float32')
    actions = np.zeros((N, 14), dtype='float32')
    for i in tqdm(range(N)):
        lightning_angle = np.load(f"transformed_data/initial_transform/lightning_angle/{i}.npy")
        lightning_gripper = np.load(f"transformed_data/initial_transform/lightning_gripper/{i}.npy")
        thunder_angle = np.load(f"transformed_data/initial_transform/thunder_angle/{i}.npy")
        thunder_gripper = np.load(f"transformed_data/initial_transform/thunder_gripper/{i}.npy")
        states[i] = np.concatenate((lightning_angle, [lightning_gripper], thunder_angle, [thunder_gripper]))

    states[:, 6] = np.where(states[:, 6] <= 50, 0, 255)
    states[:, 13] = np.where(states[:, 13] <= 50, 0, 255)
    actions[:N-1] = states[1:]
    actions[-1] = states[-1]

    data_group['states'][:] = states
    data_group['actions'][:] = actions

    for i in tqdm(range(len(episode_ends_raw))):
        start = 0
        if i > 0:
            start = episode_ends_raw[i-1] + 1
        end = episode_ends_raw[i] + 1
        images_front = [np.load(f"transformed_data/initial_transform/camera_both_front/{j}.npy") for j in range(start, end)]
        images_thunder = [np.load(f"transformed_data/initial_transform/camera_thunder_wrist/{j}.npy") for j in range(start, end)]
        images_lightning = [np.load(f"transformed_data/initial_transform/camera_lightning_wrist/{j}.npy") for j in range(start, end)]

        data_group['img_front'][start:end] = np.array(images_front)/255.0
        data_group['img_wrist_thunder'][start:end] = np.array(images_thunder)/255.0
        data_group['img_wrist_lightning'][start:end] = np.array(images_lightning)/255.0

def images_to_npy(root_path):
    ## Count number of files
    N = len(os.listdir(root_path))
    ## Check the shape of the first image
    sample_image = np.load(os.path.join(root_path, '0.npy'))

    H, W, C = sample_image.shape

    ## Create a new nparray
    data = np.zeros((N, H, W, C), dtype='float32')
    for i in tqdm(range(N)):
        data[i] = np.load(os.path.join(root_path, f'{i}.npy'))
    
    ## Normalize the data
    data = data / 255.0

    ## Save the data
    np.save(f'{root_path}.npy', data)

def save_states_and_actions():
    N = len(os.listdir("transformed_data/initial_transform/lightning_angle"))
    print(N)
    states = np.zeros((N, 14), dtype='float32')
    actions = np.zeros((N, 14), dtype='float32')

    for i in tqdm(range(N)):
        lightning_angle = np.load(f"transformed_data/initial_transform/lightning_angle/{i}.npy")
        lightning_gripper = np.load(f"transformed_data/initial_transform/lightning_gripper/{i}.npy")
        thunder_angle = np.load(f"transformed_data/initial_transform/thunder_angle/{i}.npy")
        thunder_gripper = np.load(f"transformed_data/initial_transform/thunder_gripper/{i}.npy")

        states[i] = np.concatenate((lightning_angle, [lightning_gripper], thunder_angle, [thunder_gripper]))
    
    states[:, 6] = np.where(states[:, 6] <= 50, 0, 255)
    states[:, 13] = np.where(states[:, 13] <= 50, 0, 255)
    actions[:N-1] = states[1:]
    actions[-1] = states[-1]

    np.save("transformed_data/initial_transform/states.npy", states)
    np.save("transformed_data/initial_transform/actions.npy", actions)

if __name__ == '__main__':
#     # Replace with your original dataset path
    # original_data_folder = 'raw_data'
    # transformer = TransformData(original_data_folder)
    # print(transformer.episode_ends)
#     # data2zarr()

    # images_to_npy('transformed_data/initial_transform/camera_both_front')
    # images_to_npy('transformed_data/initial_transform/camera_thunder_wrist')
    # images_to_npy('transformed_data/initial_transform/camera_lightning_wrist')

    save_states_and_actions()