#!/usr/bin/env python

import rospy
from vision.msg import plant_info

def talker():
    pub = rospy.Publisher('camera_vision', plant_info, queue_size=10)
    rospy.init_node('vision', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():


        #pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass