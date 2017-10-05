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
#include <utility>

class MoveActionServer {
public:
    MoveActionServer(ros::NodeHandle &n);

    std::string getName() const {return name;}

    void test();
private:
    const std::string name;

    std::pair<int,int> counter[3];
    const float distance;

    actionlib::SimpleActionServer<robot::MovePlantAction> as_;
    robot::MovePlantResult result;
    ScanNPlan planner;

    void exec(const robot::MovePlantGoalConstPtr &goal);

};

#endif //SMR1_MOVESERVICE_H
