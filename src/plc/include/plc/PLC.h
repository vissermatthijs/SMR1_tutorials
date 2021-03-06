//
// Created by ros-industrial on 9/20/17.
//

#ifndef SMR1_PLC_H
#define SMR1_PLC_H

#include <plc/snap7.h>
#include <ros/ros.h>
#include <vision/plant_info.h>


enum SENSORS {
    IR,
    TEST_MODE,
    MOVE_BASE
};

class PLC {
public:
    PLC(std::string ip, ros::NodeHandle& nh);
    TS7CpuInfo getCPUInfo() const;
    TS7CpInfo getCPInfo() const;

    void *getSensorValue(SENSORS s);

    void publishSensorData();

    ~PLC();

private:

    TS7Client *Client;
    int rack,slot;

    ros::Publisher sensor_pub;
    ros::Subscriber vision_sub;

    bool Check(int Result, const char * function);
    void setSkipPlant(const vision::plant_info::ConstPtr& msg);


};

#endif //SMR1_PLC_H
