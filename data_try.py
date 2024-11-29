import zarr
import cv2
import numpy as np

path = 'pusht/pusht_cchi_v7_replay.zarr'

dataset = zarr.open(path, mode='r')

print(dataset.tree())

# print(dataset['data']['img'][0])

print(np.max(dataset['data']['img'][0]))
print(np.min(dataset['data']['img'][0]))

# cv2.imshow('image', dataset['data']['img'][0]/255)
# cv2.waitKey(0)
print(dataset['meta']['episode_ends'].shape)
print(dataset['meta']['episode_ends'][0:206])