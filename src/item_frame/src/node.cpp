//
// Created by ros-industrial on 9/26/17.
//


#include <ros/ros.h>
#include <plc/sensor_info.h>
#include <item_frame/Frame.h>

#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <robot/MovePlantAction.h>
#include <vision/plant_info.h>

#include <queue>

tf::TransformBroadcaster *br = nullptr;
bool _switch = false;
bool _switchTestMode = false;
int _switchMoveBase = -1;

std::queue<int> plant_queue;

void sensorCallback(const plc::sensor_info::ConstPtr& msg)
{

    ROS_INFO("item_frame: received moveplant action");

    if((msg->ir && !_switch) || (msg->test_mode && !_switchTestMode) ||
            (msg->move_base && msg->move_base != _switchMoveBase)) {

        robot::MovePlantGoal goal;

        if(msg->test_mode) {
            _switchTestMode = true;
            goal.type = 1;
        } else if(msg->ir) {
            if(!plant_queue.empty()) {
                _switch = true;
                goal.type = plant_queue.front();
                plant_queue.pop();
            } else {
                ROS_ERROR("Error: plant queue not in sync!!");
                return;
            }
        } else if(msg->move_base) {
            goal.type = 0;
            _switchMoveBase = msg->move_base;
        } else {
            return;
        }


        ROS_INFO("item_frame: creating action client");
        static actionlib::SimpleActionClient<robot::MovePlantAction> ac("MovePlant", true);
        ac.waitForServer();

        ROS_INFO("item_frame: action client server connected. Sending goal");

        ac.sendGoal(goal);

        bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

        if (finished_before_timeout)
        {
            actionlib::SimpleClientGoalState state = ac.getState();
            ROS_INFO("Action finished: %s",state.toString().c_str());
        }
        else {
            ROS_INFO("Action did not finish before the time out.");
        }

    } else if(!msg->ir && _switch) {
        _switch = false;
    } else if(!msg->test_mode && _switchTestMode) {
        _switchTestMode = false;
    }

}

void visionCallback(const vision::plant_info::ConstPtr& msg) {
    if(msg->category > 0) {
        plant_queue.push(msg->category);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "item_frame");
    br = new tf::TransformBroadcaster();

    ros::NodeHandle nh;
    ros::NodeHandle private_node_handle ("~");

    ros::Duration(.5).sleep();  // wait for the class to initialize

    ros::Rate loop_rate(1);

    ros::Subscriber sub = nh.subscribe("plc_sensors", 1000, sensorCallback);
    ros::Subscriber sub2 = nh.subscribe("vision", 1000, visionCallback);

    while (ros::ok()) {

        ros::spinOnce();
        loop_rate.sleep();
    }

    ros::waitForShutdown();
    delete br;
}