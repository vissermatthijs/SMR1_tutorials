//
// Created by ros-industrial on 9/13/17.
//

#include <robot/ScanNPlan.h>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/solid_primitive_dims.h>

ScanNPlan::ScanNPlan(ros::NodeHandle &nh, bool useConstraints) : move_group("manipulator") {

   /* if(useConstraints) {
        this->ocm.link_name = "eoat";
        this->ocm.header.frame_id = "tool0";
        this->ocm.absolute_x_axis_tolerance = 0.2;
        this->ocm.absolute_y_axis_tolerance = 0.2;
        this->ocm.absolute_z_axis_tolerance = 0.2;
        this->ocm.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0f, 0.0f , 6.28f);

        this->constraints.orientation_constraints.push_back(ocm);
        this->move_group.setPathConstraints(this->constraints);
    } */

    this->move_group.setEndEffectorLink("tool0");
    this->plant_object.header.frame_id = this->move_group.getPlanningFrame();
    this->plant_object.id = "plant";
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.CYLINDER;

    primitive.dimensions.resize(
            geometric_shapes::SolidPrimitiveDimCount<shape_msgs::SolidPrimitive::CYLINDER>::value);
    primitive.dimensions[primitive.CYLINDER_HEIGHT] = 0.20;
    primitive.dimensions[primitive.CYLINDER_RADIUS] = 0.025;

    geometry_msgs::Pose object_pose;
    object_pose.orientation.w = 1.0;
    object_pose.position.x = -0.30;
    object_pose.position.y = -0.18;
    object_pose.position.z = 1.13;


    this->plant_object.primitives.push_back(primitive);
    this->plant_object.primitive_poses.push_back(object_pose);
    this->plant_object.operation = this->plant_object.ADD;
    this->collision_objects.push_back(this->plant_object);

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

void ScanNPlan::manualPose(geometry_msgs::Pose &p) {

    move_group.setPoseTarget(p);
    move_group.move();
}

void ScanNPlan::pushConstraintFromCurrentOrientation() {

    this->ocm.link_name = "tool0";
    this->ocm.weight = 1.0;
    this->ocm.orientation = this->getCurrentOrientation();
    this->ocm.absolute_x_axis_tolerance = 3.14 / 2.0;
    this->ocm.absolute_y_axis_tolerance = 3.14 / 2.0;
    this->ocm.absolute_z_axis_tolerance = 3.14 * 2.0;

    this->constraints.orientation_constraints.push_back(ocm);
    this->move_group.setPathConstraints(this->constraints);

}

void ScanNPlan::popCurrentConstraint() {

    this->constraints.orientation_constraints.clear();
    this->move_group.clearPathConstraints();
}

void ScanNPlan::pushCollisionObject(){

    //this->plant_object.header.frame_id = this->move_group.getPlanningFrame();
    this->planning_scene_interface.addCollisionObjects(this->collision_objects);
    ros::Duration(1.0).sleep();
    this->move_group.attachObject(this->plant_object.id,"eoat");
}

void ScanNPlan::popCollisionObject(){
    this->move_group.detachObject(this->plant_object.id);

    std::vector<std::string> ids;
    ids.push_back(this->plant_object.id);
    this->planning_scene_interface.removeCollisionObjects(ids);

}

std::map<std::string, double> ScanNPlan::getNamedTarget(std::string t){
    return this->move_group.getNamedTargetValues(t);
};