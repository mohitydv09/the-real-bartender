import numpy as np
import matplotlib.pyplot as plt

action = np.load('action.npy')

stats = np.load('dataset/transformed_data/uncork_v2_stats.npy', allow_pickle=True).item()


print(action.shape)

plt.plot(action[:, 0])
plt.plot(action[:, 1])
plt.plot(action[:, 2])
plt.plot(action[:, 3])
plt.plot(action[:, 4])
plt.plot(action[:, 5])

plt.show()