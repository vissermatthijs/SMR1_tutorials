#include <ros/ros.h>
#include <plc/PLC.h>
#include <iostream>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "plc");

    ros::NodeHandle nh;
    ros::NodeHandle private_node_handle ("~");

    std::string plc_ip;
    private_node_handle.param<std::string>("plc_ip", plc_ip, "192.168.0.10"); // parameter name, string object reference, default value

    ros::Duration(.5).sleep();  // wait for the class to initialize

    ros::Rate loop_rate(1);

    PLC plc(plc_ip, nh);

    while (ros::ok()) {

        plc.publishSensorData();

        ros::spinOnce();
        loop_rate.sleep();
    }

    ros::waitForShutdown();
}