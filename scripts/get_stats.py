import zarr 
import os
import numpy as np

# dataset_path = 'uncork_v2.zarr'

# dataset_root = zarr.open(dataset_path, mode='r')

# data_set_path = 'data_mohit'
# print(dataset_root.tree())
# N = len(os.listdir('raw_data'))
# states = np.zeros((N, 14))
# actions = np.zeros((N, 14))
# for filename in os.listdir("raw_data"):
#     data_ = np.load(os.path.join('raw_data', filename))


def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

states = np.load('v5_50/states.npy')
actions = np.load('v5_50/actions.npy')

train_data = {
    'agent_pos': states, ## (N,14)
    'action': actions    ## (N,14)
}

# compute statistics and normalized data to [-1,1]
stats = dict()
normalized_train_data = dict()
for key, data in train_data.items():
    stats[key] = get_data_stats(data)

print("Stats: ", stats)

np.save('uncork_v5_stats.npy', stats)