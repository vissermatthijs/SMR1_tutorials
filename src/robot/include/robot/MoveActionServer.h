//
// Created by ros-industrial on 9/13/17.
//

#ifndef SMR1_MOVESERVICE_H
#define SMR1_MOVESERVICE_H

#include <ros/ros.h>
#include <robot/ScanNPlan.h>
#include <actionlib/server/simple_action_server.h>
#include <robot/MovePlantAction.h>
#include <string>

class MoveActionServer {
public:
    MoveActionServer(ros::NodeHandle &n);

    std::string getName() const {return name;}

private:
    const std::string name;

    actionlib::SimpleActionServer<robot::MovePlantAction> as_;
    ScanNPlan planner;

    void exec(const robot::MovePlantGoalConstPtr &goal);

};

#endif //SMR1_MOVESERVICE_H
