//
// Created by ros-industrial on 9/13/17.
//

#ifndef SMR1_SCANNPLAN_H
#define SMR1_SCANNPLAN_H

#ifndef ROBOTLAB_WS_SCNANNPLAN_H
#define ROBOTLAB_WS_SCNANNPLAN_H

#include <ros/ros.h>
#include <tf/tf.h>
#include <moveit/move_group_interface/move_group_interface.h>


class ScanNPlan
{
public:
    ScanNPlan(ros::NodeHandle& nh);
    void randomPoses();
    void manualPose(const std::string& pose);
    void manualPose(float x, float y, float z);

    bool plan(float x, float y, float z);

    geometry_msgs::PoseStamped getCurrentPose() {
        return this->move_group.getCurrentPose();
    }

private:

    moveit::planning_interface::MoveGroupInterface move_group;
    moveit_msgs::OrientationConstraint ocm;
    moveit_msgs::Constraints constraints;
};

#endif //ROBOTLAB_WS_SCNANNPLAN_H


#endif //SMR1_SCANNPLAN_H
