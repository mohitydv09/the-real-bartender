import rospy
import sys
import numpy as np
sys.path.append("/home/rpmdt05/Code/the-real-bartender/Spark/TeleopMethods")

from Spark.TeleopMethods.UR.arms import UR
import rospy
from std_msgs.msg import Float32MultiArray 
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from collections import deque

import rtde_receive

def robot_state_publisher(lightning_reciver, thunder_reciver, pubs):
    lightning_q = lightning_reciver.getActualQ()
    pubs['Lightning_q'].publish(Float32MultiArray(data=lightning_q))
    thunder_q = thunder_reciver.getActualQ()
    pubs['Thunder_q'].publish(Float32MultiArray(data=thunder_q))
    lightning_gripper = 3.0
    pubs['Lightning_gripper'].publish(Float32(data=lightning_gripper))
    thunder_gripper = 3.0
    pubs['Thunder_gripper'].publish(Float32(data=thunder_gripper))

if __name__=='__main__':
    rospy.init_node("robot_state_publisher")
    thunder_ip = "192.168.0.101"
    lightning_ip = "192.168.0.102"
    arms = ['Thunder', 'Lightning']

    lightning_receiver = rtde_receive.RTDEReceiveInterface(lightning_ip)
    thunder_receiver = rtde_receive.RTDEReceiveInterface(thunder_ip)

    pubs = dict()
    for arm in arms:
        pubs[arm+"_q"] = rospy.Publisher(f"/{arm.lower()}_q", Float32MultiArray, queue_size=10)
        pubs[arm+"_gripper"] = rospy.Publisher(f"/{arm.lower()}_gripper", Float32, queue_size=10)

    while not rospy.is_shutdown():
        robot_state_publisher(lightning_receiver, thunder_receiver, pubs)
        rospy.sleep(0.001)
    
