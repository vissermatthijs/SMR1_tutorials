#!/usr/bin/env python
import rospy
from vision.msg import plant_info

import actionlib
import robot.msg

client = None

def callAction(_type, _x, _y, _z):
    goal = robot.msg.MovePlantGoal(type=_type, x=_x, y=_y, z=_z)

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()


def callback(data):
    print data

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('learning', anonymous=True)

    client = actionlib.SimpleActionClient('MovePlant', robot.msg.MovePlantAction)
    client.wait_for_server()

    rospy.Subscriber("camera_vision", plant_info, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()