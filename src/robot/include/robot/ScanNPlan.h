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

private:

    moveit::planning_interface::MoveGroupInterface move_group;
};

#endif //ROBOTLAB_WS_SCNANNPLAN_H


#endif //SMR1_SCANNPLAN_H
