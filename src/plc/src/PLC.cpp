//
// Created by ros-industrial on 9/20/17.
//

#include <plc/PLC.h>
#include <iostream>
#include <ros/ros.h>
#include <plc/sensor_info.h>

PLC::PLC(std::string ip, ros::NodeHandle& nh) : rack(0), slot(1) {

    Client= new TS7Client();

    int res = Client->ConnectTo(ip.c_str(),rack,slot);
    if (Check(res,"UNIT Connection")) {
        printf("  Connected to   : %s (Rack=%d, Slot=%d)\n",ip.c_str(),rack,slot);
        printf("  PDU Requested  : %d bytes\n",Client->PDURequested());
        printf("  PDU Negotiated : %d bytes\n",Client->PDULength());
    };


    if(res!=0) {
        ROS_INFO("PLC Connection error");
    }

    sensor_pub = nh.advertise<plc::sensor_info>("plc_sensors", 1000);
}

PLC::~PLC() {
    Client->Disconnect();
}

bool PLC::Check(int Result, const char * function) {

   /* printf("\n");
    printf("+-----------------------------------------------------\n");
    printf("| %s\n",function);
    printf("+-----------------------------------------------------\n"); */
    if (Result==0) {
       /* printf("| Result         : OK\n");
        printf("| Execution time : %d ms\n",Client->ExecTime());
        printf("+-----------------------------------------------------\n"); */
    }
    else {
        printf("| ERROR !!! \n");
        if (Result<0)
            printf("| Library Error (-1)\n");
        else
            printf("| %s\n",CliErrorText(Result).c_str());
        printf("+-----------------------------------------------------\n");
    }
    return Result==0;


}

bool PLC::getDISensorValue(SENSORS s) {


    int start;
    byte EB;

    switch(s) {
        case SENSORS::IR:
            start = 0;
            break;
        default:
            return -1;
    }

    int res=Client->ReadArea(S7AreaPE, 0, start, 1, S7WLByte, &EB);

    if (Check(res,"ReadArea"))
    {
        return static_cast<bool>(EB);
    } else {
        return -1;
    }
}

void PLC::publishSensorData() {

    plc::sensor_info msg;

    byte IR = this->getDISensorValue(SENSORS::IR);
    msg.ir = static_cast<bool>(IR);

    sensor_pub.publish(msg);
}