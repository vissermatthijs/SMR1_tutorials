//
// Created by ros-industrial on 9/13/17.
//

#include <robot/MoveActionServer.h>

MoveActionServer::MoveActionServer(ros::NodeHandle &n) : planner(n),
                                                         name("MovePlant"),
                                                         as_(n, name,
                                                             boost::bind(&MoveActionServer::exec, this, _1),
                                                             false) {

    as_.start();
}


void MoveActionServer::exec(const robot::MovePlantGoalConstPtr &goal) {



}

void MoveActionServer::test() {
    planner.randomPoses();
}