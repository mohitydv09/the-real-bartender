import cv2
import numpy as np

import matplotlib.pyplot as plt

episode = np.load("/home/rpmdt05/Code/the-real-bartender/dataset/raw_data/transforms_98.npy", allow_pickle=True).item()

# print(episode.keys())
# lightning_angle = np.zeros((len(episode), 6))
# for i, frame in enumerate(episode):
#     lightning_angle[i] = episode[frame]["lightning_angle"]

# lightning_gripper = np.zeros((len(episode), 1))
# for i, frame in enumerate(episode):
#     print(episode[frame]["lightning_gripper"])
#     lightning_gripper[i] = episode[frame]["lightning_gripper"]

# print(lightning_angle.shape)

# lightning_angles = np.hstack((lightning_angle, lightning_gripper))
# print(lightning_angles.shape)

# for i in range(6):
#     plt.plot(lightning_angles[:, i])
# plt.legend([1,2,3,4,5,6,7])
# plt.show()



for frame in episode:
    image = episode[frame]["camera_both_front"]
    # print(type(image))
    # print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera Thunder Wrist", image)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()