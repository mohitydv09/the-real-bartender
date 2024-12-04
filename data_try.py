#!/usr/bin/env python

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Float32MultiArray, Bool

# Callback to process synchronized messages
def sync_callback(msg1, msg2):
    rospy.loginfo("Synchronized messages received")
    pub_sync_1.publish(msg1)
    pub_sync_2.publish(msg2)

if __name__ == '__main__':
    rospy.init_node('synchronizer_node')

    # Subscribers for the topics to be synchronized
    sub1 = Subscriber('/Force/force_ctl', Float32MultiArray)
    sub2 = Subscriber('/Force/policy', Float32MultiArray)

    # Publishers for synchronized topics
    pub_sync_1 = rospy.Publisher('/sync/force_ctl', Float32MultiArray, queue_size=10)
    pub_sync_2 = rospy.Publisher('/sync/policy', Float32MultiArray, queue_size=10)

    # Synchronizer with a small time slop
    sync = ApproximateTimeSynchronizer([sub1, sub2], queue_size=10, slop=0.05)
    sync.registerCallback(sync_callback)

    rospy.loginfo("Synchronization node started")
    rospy.spin()
