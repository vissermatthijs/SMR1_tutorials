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
#include <map>
#include <string>
#include <tf/transform_listener.h>
#include <eigen_conversions/eigen_msg.h>

class ScanNPlan {
public:
    ScanNPlan(ros::NodeHandle &nh, bool useConstraints = true);

    void randomPoses();

    void manualPose(const std::string &pose);

    void manualPose(float x, float y, float z, float w);

    void manualPose(float x, float y, float z, float rx, float ry, float rz);

    void manualPose(geometry_msgs::Pose &p);

    void pushConstraintFromCurrentOrientation();

    void popCurrentConstraint();

    std::map<std::string, double> getNamedTarget(std::string t);

    bool plan(float x, float y, float z);

    geometry_msgs::Pose getCurrentPose() {
        return this->move_group.getCurrentPose().pose;
    }

    geometry_msgs::Quaternion getCurrentOrientation() {

        /*robot_state::RobotState kinematic_state(*(this->move_group.getCurrentState()));

        kinematic_state.update();

        Eigen::Vector3d v = kinematic_state.getGlobalLinkTransform(
                kinematic_state.getLinkModel("tool0")).rotation().eulerAngles(0,1,2);

        return tf::createQuaternionMsgFromRollPitchYaw(v.x(),v.y(), v.z());*/

        auto current_pose = this->move_group.getCurrentPose(this->ocm.link_name);
        this->ocm.header.frame_id = current_pose.header.frame_id;

        return current_pose.pose.orientation;
    }

private:

    moveit::planning_interface::MoveGroupInterface move_group;
    moveit_msgs::OrientationConstraint ocm;
    moveit_msgs::Constraints constraints;
    tf::TransformListener listener;
};

#endif //ROBOTLAB_WS_SCNANNPLAN_H


#endif //SMR1_SCANNPLAN_H
