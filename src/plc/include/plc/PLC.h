//
// Created by ros-industrial on 9/20/17.
//

#ifndef SMR1_PLC_H
#define SMR1_PLC_H

#include <plc/snap7.h>
#include <ros/ros.h>



enum SENSORS {
    IR
};

class PLC {
public:
    PLC(std::string ip, ros::NodeHandle& nh);
    TS7CpuInfo getCPUInfo() const;
    TS7CpInfo getCPInfo() const;

    bool getDISensorValue(SENSORS s);

    ~PLC();

private:

    TS7Client *Client;
    int rack,slot;

    ros::Publisher sensor_pub;

    bool Check(int Result, const char * function);


};

#endif //SMR1_PLC_H
