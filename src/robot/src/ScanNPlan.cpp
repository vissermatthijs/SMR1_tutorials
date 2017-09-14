//
// Created by ros-industrial on 9/13/17.
//

#include <robot/ScanNPlan.h>

ScanNPlan::ScanNPlan(ros::NodeHandle &nh) : move_group("manipulator") {
}

void ScanNPlan::randomPoses() {

    bool success = false;

    while(!success) {

        ROS_INFO("Calculating new Pose!!");

        move_group.setPoseTarget(move_group.getRandomPose());
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        success = move_group.plan(my_plan);
    }

    ROS_INFO("Found good pose, moving...");
    move_group.move();
}

void ScanNPlan::manualPose(const std::string &pose) {

    std::string s("Going to pose: " + pose);
    ROS_INFO(s.c_str());
    ROS_INFO("Moving...");
    move_group.setNamedTarget(pose);
    move_group.move();
}


void ScanNPlan::manualPose(float x, float y, float z) {

    geometry_msgs::Pose move_target;
    move_target.orientation.w = 1.0;
    move_target.position.x = x;
    move_target.position.y = y;
    move_target.position.z = z;

    move_group.setPoseTarget(move_target);
    move_group.move();
}