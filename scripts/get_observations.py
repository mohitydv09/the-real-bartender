import rospy
from std_msgs.msg import Float32MultiArray 
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import numpy as np

import config_.yaml


'''
This class is used for getting the observations from the robot based on the observation horizon

'''

class ObservationSubscriber:
    def __init__(self):
        self.node_name = 'observation_subscriber'
        rospy.init_node(self.node_name, anonymous=True)
        self.lightning_angle = np.array([])
        self.lightning_gripper_ = np.array([])

        self.thunder_angle = np.array([])
        self.thunder_gripper_ = np.array([])

        self.image_both_front = np.array([])
        self.image_lightning_wrist = np.array([])
        self.image_thunder_wrist = np.array([])


        rospy.Subscriber('/cameras/rgb/thunder/wrist', Image, self.thunder_wrist)
        rospy.Subscriber('/cameras/rgb/lightning/wrist', Image, self.lightning_wrist)
        rospy.Subscriber('cameras/rgb/both/front', Image, self.both_front)

        rospy.Subscriber('/lightning_q', Float32MultiArray, self.lightning_q)
        rospy.Subscriber('/thunder_q', Float32MultiArray, self.thunder_q)

        rospy.Subscriber('/thunder_gripper', Float32, self.thunder_gripper)
        rospy.Subscriber('/lightning_gripper', Float32, self.lightning_gripper)

    def thunder_wrist(self, data):
        self.image_thunder_wrist = data

