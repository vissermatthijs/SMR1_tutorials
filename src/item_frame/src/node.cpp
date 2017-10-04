//
// Created by ros-industrial on 9/26/17.
//


#include <ros/ros.h>
#include <plc/sensor_info.h>
#include <item_frame/Frame.h>

#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <robot/MovePlantAction.h>

tf::TransformBroadcaster *br = nullptr;
bool _switch = false;

void sensorCallback(const plc::sensor_info::ConstPtr& msg)
{

    ROS_INFO("item_frame: received moveplant action");

    if(msg->ir && !_switch) {

        _switch = true;
        ROS_INFO("item_frame: creating action client");
        static actionlib::SimpleActionClient<robot::MovePlantAction> ac("MovePlant", true);
        ac.waitForServer();

        ROS_INFO("item_frame: action client server connected. Sending goal");

        robot::MovePlantGoal goal;
        goal.x = 0.0f;
        goal.y = 0.0f;
        goal.z = 0.0f;
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
    } else {
        ROS_INFO("item_frame: ir value is false");
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

    while (ros::ok()) {

        ros::spinOnce();
        loop_rate.sleep();
    }

    ros::waitForShutdown();
    delete br;
}