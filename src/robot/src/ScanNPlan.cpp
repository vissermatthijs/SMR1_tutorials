//
// Created by ros-industrial on 9/13/17.
//

#include <robot/ScanNPlan.h>

ScanNPlan::ScanNPlan(ros::NodeHandle &nh) : move_group("manipulator") {

    this->ocm.link_name = "eoat";
    this->ocm.header.frame_id = "base_link";
    this->ocm.orientation.w = 1.0;
    this->ocm.absolute_x_axis_tolerance = 1.0;
    this->ocm.absolute_y_axis_tolerance = 1.0;
    this->ocm.absolute_z_axis_tolerance = 1.0;
    this->ocm.weight = 1.0;

    this->constraints.orientation_constraints.push_back(ocm);
    //this->move_group.setPathConstraints(this->constraints);
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
    move_target.position.x = x;
    move_target.position.y = y;
    move_target.position.z = z;

    move_group.setPoseTarget(move_target);
    move_group.move();
}

bool ScanNPlan::plan(float x, float y, float z) {

    geometry_msgs::Pose move_target;
    move_target.position.x = x;
    move_target.position.y = y;
    move_target.position.z = z;

    move_group.setPoseTarget(move_target);
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    return move_group.plan(my_plan);
};