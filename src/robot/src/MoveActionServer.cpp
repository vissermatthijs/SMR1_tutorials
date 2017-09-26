//
// Created by ros-industrial on 9/13/17.
//

#include <robot/MoveActionServer.h>

MoveActionServer::MoveActionServer(ros::NodeHandle &n) : planner(n),
                                                         name("MovePlant"),
                                                         as_(n, name,
                                                             boost::bind(&MoveActionServer::exec, this, _1),
                                                             false) {

    this->as_.start();
}


void MoveActionServer::exec(const robot::MovePlantGoalConstPtr &goal) {

    bool gotPath = this->planner.plan(goal->x, goal->y, goal->z);

    if(gotPath) {
        this->planner.manualPose(goal->x, goal->y, goal->z);

        this->result.sequence = true;
        this->as_.setSucceeded(this->result);

    } else {
        this->as_.setPreempted();
    }

}

void MoveActionServer::test() {
    this->planner.randomPoses();
}