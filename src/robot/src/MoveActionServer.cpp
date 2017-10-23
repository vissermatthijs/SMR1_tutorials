//
// Created by ros-industrial on 9/13/17.
//

#include <robot/MoveActionServer.h>
#include <string>
#include <sstream>

MoveActionServer::MoveActionServer(ros::NodeHandle &n) : planner(n, false),
                                                         name("MovePlant"),
                                                         as_(n, name,
                                                             boost::bind(&MoveActionServer::exec, this, _1),
                                                             false), distance(0.115f) {

    this->as_.start();

    counter[0].first = 1;
    counter[0].second = 0;

    counter[1].first = 0;
    counter[1].second = 0;

    counter[2].first = 0;
    counter[2].second = 0;
}


void MoveActionServer::exec(const robot::MovePlantGoalConstPtr &goal) {

    if(goal->type == 0) {
        this->planner.manualPose("home");
    } else if(goal->type == 1) {
        this->planner.manualPose("pickup_step1");
        this->planner.manualPose("pickup_step2");
        this->planner.manualPose("pickup_step3");

        this->planner.pushCollisionObject();
        //this->planner.pushConstraintFromCurrentOrientation();
        this->planner.manualPose("place_bin1");
        //this->planner.popCurrentConstraint();



        // Get current pose
        geometry_msgs::Pose p = this->planner.getCurrentPose();

        // Move to right place (xy)
        p.position.y -= counter[0].first * distance;
        p.position.x -= counter[0].second * distance;
        this->planner.manualPose(p);

        // Place plant (z)
        p.position.z -= 0.24f;
        this->planner.manualPose(p);

        // Remove end effector from plant
        p.position.x -= 0.15f;
        p.position.y -= 0.15f;
        this->planner.manualPose(p);

        counter[0].first++;

        if(counter[0].first > 4) {
            counter[0].first = 0;
            counter[0].second = 1;
        }
      //  this->planner.popCollisionObject();
    }

    this->result.sequence = true;
    this->as_.setSucceeded(this->result);
}

void MoveActionServer::test() {
  //  this->planner.manualPose(-0.214f, -0.245f, 1.84f, -1.0f);

  /*  std::stringstream ss;
    ss << "[Position] X: " << this->planner.getCurrentPose().pose.position.x << " Y: " <<
       this->planner.getCurrentPose().pose.position.y << " Z: " << this->planner.getCurrentPose().pose.position.z;

    ROS_INFO(ss.str().c_str());

    std::stringstream ss2;
    ss2 << "[Orientation] X: " << this->planner.getCurrentPose().pose.orientation.x << " Y: " <<
       this->planner.getCurrentPose().pose.orientation.y << " Z: " << this->planner.getCurrentPose().pose.orientation.z <<
        " W: " << this->planner.getCurrentPose().pose.orientation.w;

    ROS_INFO(ss2.str().c_str()); */
}

