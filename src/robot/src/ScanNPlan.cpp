//
// Created by ros-industrial on 9/13/17.
//

#include <robot/ScanNPlan.h>
#include <tf/tf.h>

ScanNPlan::ScanNPlan(ros::NodeHandle &nh, bool useConstraints) : move_group("manipulator") {

    if(useConstraints) {
        this->ocm.link_name = "eoat";
        this->ocm.header.frame_id = "tool0";
        this->ocm.absolute_x_axis_tolerance = 0.2;
        this->ocm.absolute_y_axis_tolerance = 0.2;
        this->ocm.absolute_z_axis_tolerance = 0.2;
      //  this->ocm.orientation = this->move_group.getPoseTarget("pickup_step2").orientation;

        this->constraints.orientation_constraints.push_back(ocm);
        this->move_group.setPathConstraints(this->constraints);
    }
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


void ScanNPlan::manualPose(float x, float y, float z, float w) {

    geometry_msgs::Pose move_target;
    move_target.position.x = x;
    move_target.position.y = y;
    move_target.position.z = z;
    move_target.orientation.w = w;

    move_group.setPoseTarget(move_target);
    move_group.move();
}

void ScanNPlan::manualPose(float x, float y, float z, float rx, float ry, float rz) {
    geometry_msgs::Pose move_target;
    move_target.position.x = x;
    move_target.position.y = y;
    move_target.position.z = z;

    geometry_msgs::Quaternion q = tf::createQuaternionMsgFromRollPitchYaw(rx, ry,rz);
    move_target.orientation = q;

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

std::map<std::string, double> ScanNPlan::getNamedTarget(std::string t){
    return this->move_group.getNamedTargetValues(t);
};