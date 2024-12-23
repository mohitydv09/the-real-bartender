import os
import zarr
import numpy as np

## Read the data from the directory
raw_data_directory = 'transformed_data/initial_transform'

num_of_data_points = len(os.listdir(os.path.join(raw_data_directory, 'camera_thunder_wrist')))
print("Number of data points: ", num_of_data_points)
N = num_of_data_points
image_shape = np.load("transformed_data/initial_transform/camera_both_front/0.npy").shape
H, W, C = image_shape

## Create a new Zarr Group
zarr_data_file_path = 'uncork_v2.zarr'
root = zarr.open_group(zarr_data_file_path, mode='w')

## Create data group
data_group = root.create_group('data')

data_group.create_dataset('img_front', shape=(N, H, W, C), chunks=(1, H, W, C), dtype='float32')
data_group.create_dataset('img_wrist_thunder', shape=(N, H, W, C), dtype='float32')
data_group.create_dataset('img_wrist_lightning', shape=(N, H, W, C), dtype='float32')
data_group.create_dataset('states', shape=(N,14), dtype='float32')    ## UR angles
data_group.create_dataset('actions', shape=(N,14), dtype='float32')   ## Spark Angles

meta_group = root.create_group('meta')

## Fill the data
episode_ends_raw = np.load("transformed_data/initial_transform/episode_ends.npy")
meta_group.create_dataset('episode_ends', shape=episode_ends_raw.shape, chunks=episode_ends_raw.shape, dtype='int64')
meta_group['episode_ends'][:] = episode_ends_raw

## Front Camera Files.
for i in range(N):
    print(f"Processing {i}/{N}")
    image_front = np.load(f"transformed_data/initial_transform/camera_both_front/{i}.npy")
    lightning_wrist = np.load(f"transformed_data/initial_transform/camera_lightning_wrist/{i}.npy")
    thunder_wrist = np.load(f"transformed_data/initial_transform/camera_thunder_wrist/{i}.npy")
    data_group['img_front'][i] = image_front/255.0
    data_group['img_wrist_lightning'][i] = lightning_wrist/255.0
    data_group['img_wrist_thunder'][i] = thunder_wrist/255.0
    
    lightning_angles = np.load(f"transformed_data/initial_transform/lightning_angle/{i}.npy")
    lightning_gripper = np.load(f"transformed_data/initial_transform/lightning_gripper/{i}.npy")

    thunder_angles = np.load(f"transformed_data/initial_transform/thunder_angle/{i}.npy")
    thunder_gripper = np.load(f"transformed_data/initial_transform/thunder_gripper/{i}.npy")

    data_group['states'][i] = np.concatenate((lightning_angles, [lightning_gripper], thunder_angles, [thunder_gripper]))

    spark_lightning_angles = np.load(f"transformed_data/initial_transform/spark_lightning_angle/{i}.npy")
    spark_thunder_angles = np.load(f"transformed_data/initial_transform/spark_thunder_angle/{i}.npy")

    data_group['actions'][i] = np.concatenate((spark_lightning_angles, spark_thunder_angles))