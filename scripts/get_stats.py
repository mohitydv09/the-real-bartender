import numpy as np

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