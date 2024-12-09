import rospy
from std_msgs.msg import Float32MultiArray 
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import numpy as np
from collections import deque

'''
This class is used for getting the observations from the robot based on the observation horizon.
each observation is a dictionary with the following keys:
- robot_pos: [lightning_angle, lightning_gripper, thunder_angle, thunder_gripper] (concatenation of the 4 keys)
- thunder_gripper: the gripper of the thunder robot
- image_both_front: the image from the front camera
- image_lightning_wrist: the image from the lightning robot wrist camera
- image_thunder_wrist: the image from the thunder robot wrist camera

The objects are stores in a queue format, where the oldest observation is removed when a new one is added.
The queue is of size observation_horizon, which is a parameter in the config.yaml file.
'''

class ObservationSubscriber:
    def __init__(self, obs_horizon):
        self.node_name = 'image_subscriber'
        rospy.init_node(self.node_name, anonymous=True)

        # Define the observation horizon
        self.obs_horizon = obs_horizon

        # Deques to store the last 'n' observations for each image
        self.img_front_history = deque(maxlen=obs_horizon)
        self.img_wrist_thunder_history = deque(maxlen=obs_horizon)
        self.img_wrist_lightning_history = deque(maxlen=obs_horizon)

        # Deques for agent states
        self.lightning_angle_history = deque(maxlen=obs_horizon)
        self.thunder_angle_history = deque(maxlen=obs_horizon)
        self.lightning_gripper_history = deque(maxlen=obs_horizon)
        self.thunder_gripper_history = deque(maxlen=obs_horizon)

        # Temporary storage for incoming messages
        self.latest_img_front = None
        self.latest_img_wrist_thunder = None
        self.latest_img_wrist_lightning = None
        self.latest_lightning_angle = None
        self.latest_thunder_angle = None
        self.latest_lightning_gripper = None
        self.latest_thunder_gripper = None

        # Subscribers
        rospy.Subscriber('/cameras/rgb/thunder/wrist', Image, self.thunder_wrist)
        rospy.Subscriber('/cameras/rgb/lightning/wrist', Image, self.lightning_wrist)
        rospy.Subscriber('cameras/rgb/both/front', Image, self.both_front)
        rospy.Subscriber('/lightning_q', Float32MultiArray, self.lightning_q)
        rospy.Subscriber('/thunder_q', Float32MultiArray, self.thunder_q)
        rospy.Subscriber('/thunder_gripper', Float32, self.thunder_gripper)
        rospy.Subscriber('/lightning_gripper', Float32, self.lightning_gripper)


    def thunder_wrist(self, msg):
        # Store the latest wrist image for thunder
        self.latest_img_wrist_thunder = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        # Append all queued data to history when the thunder wrist image arrives
        if self.latest_img_wrist_thunder is not None:
            self.img_wrist_thunder_history.append(self.latest_img_wrist_thunder)
            if self.latest_img_front is not None:
                self.img_front_history.append(self.latest_img_front)
            if self.latest_img_wrist_lightning is not None:
                self.img_wrist_lightning_history.append(self.latest_img_wrist_lightning)
            if self.latest_lightning_angle is not None:
                self.lightning_angle_history.append(self.latest_lightning_angle)
            if self.latest_thunder_angle is not None:
                self.thunder_angle_history.append(self.latest_thunder_angle)
            if self.latest_lightning_gripper is not None:
                self.lightning_gripper_history.append(self.latest_lightning_gripper)
            if self.latest_thunder_gripper is not None:
                self.thunder_gripper_history.append(self.latest_thunder_gripper)

    def lightning_wrist(self, msg):
        # Store the latest lightning wrist image
        self.latest_img_wrist_lightning = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    def both_front(self, msg):
        # Store the latest front image
        self.latest_img_front = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    def lightning_q(self, msg):
        # Store the latest lightning angle (size 6)
        self.latest_lightning_angle = np.asarray(msg.data)

    def thunder_q(self, msg):
        # Store the latest thunder angle (size 6)
        self.latest_thunder_angle = np.asarray(msg.data)

    def lightning_gripper(self, msg):
        # Store the latest lightning gripper state (scalar)
        self.latest_lightning_gripper = np.asarray(msg.data)

    def thunder_gripper(self, msg):
        # Store the latest thunder gripper state (scalar)
        self.latest_thunder_gripper = np.asarray(msg.data)

    def get_last_n_observations(self):
        """Returns the last 'n' observations structured as per the requirement."""
        images = {
            'img_front': np.array(self.img_front_history),
            'img_wrist_thunder': np.array(self.img_wrist_thunder_history),
            'img_wrist_lightning': np.array(self.img_wrist_lightning_history),
        }

        # Agent state is a concatenation of:
        # - lightning angle (6,)
        # - lightning gripper (1,)
        # - thunder angle (6,)
        # - thunder gripper (1,)
        agent_state = np.column_stack((
            np.array(self.lightning_angle_history),        # (obs_horizon, 6)
            np.array(self.lightning_gripper_history),      # (obs_horizon, 1)
            np.array(self.thunder_angle_history),         # (obs_horizon, 6)
            np.array(self.thunder_gripper_history)        # (obs_horizon, 1)
        ))

        return {
            'Images': images,  # (obs_horizon, h, w, c)
            'agent_state': agent_state  # (obs_horizon, 14) - concatenation of the agent's states
        }

if __name__ == '__main__':
    try:
        image_subscriber = ObservationSubscriber(obs_horizon=10)  # You can set the horizon to whatever value you want
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node terminated.")


