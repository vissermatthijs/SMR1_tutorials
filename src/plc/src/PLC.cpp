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
        printf("| Result         : OK\n");
        printf("| Execution time : %d ms\n",Client->ExecTime());
        printf("+-----------------------------------------------------\n");
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

void *PLC::getSensorValue(SENSORS s) {


    int start;
    void *EB;
    byte area;

    int size;

    switch(s) {
        case SENSORS::IR:
            start = 5;
            area = S7AreaPE;
            size = S7WLBit;
            break;
        case SENSORS::TEST_MODE:
            start = 6;
            area = S7AreaMK;
            size = S7WLBit;
            break;
        case SENSORS::MOVE_BASE:
            start = 10;
            area = S7AreaMK;
            size = S7WLWord;
            break;
        default:
            return nullptr;
    }

    int res=Client->ReadArea(area, 0, start, 1, size, &EB);

    if (Check(res,"ReadArea"))
    {
        return EB;
    }
    else {
        return nullptr;
    }
}

void PLC::publishSensorData() {

    plc::sensor_info msg;
    msg.ir = static_cast<bool>(this->getSensorValue(SENSORS::IR));
    msg.test_mode = static_cast<bool>(this->getSensorValue(SENSORS::TEST_MODE));
    msg.move_base = reinterpret_cast<long>(this->getSensorValue(SENSORS::MOVE_BASE));

    sensor_pub.publish(msg);
}