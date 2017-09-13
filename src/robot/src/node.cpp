//
// Created by ros-industrial on 9/13/17.
//

#include <robot/MoveActionServer.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "robot");
    ros::AsyncSpinner async_spinner(1);
    async_spinner.start();

    ros::NodeHandle nh;
    ros::NodeHandle private_node_handle ("~");

    std::string base_frame;
    private_node_handle.param<std::string>("base_frame", base_frame, "world"); // parameter name, string object reference, default value

    MoveActionServer app(nh);

    ros::Duration(.5).sleep();  // wait for the class to initialize

    ros::Rate loop_rate(1);

    while (ros::ok()) {

        ros::spinOnce();
        loop_rate.sleep();
    }

    ros::waitForShutdown();
}