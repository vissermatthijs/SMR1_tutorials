//
// Created by ros-industrial on 9/20/17.
//

#include <plc/PLC.h>
#include <iostream>
#include <ros/ros.h>

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

 //sensor_pub = nh.advertise<plc::sensor_info>("plc_sensors", 1000);
}

PLC::~PLC() {
    Client->Disconnect();
}

bool PLC::Check(int Result, const char * function) {

    printf("\n");
    printf("+-----------------------------------------------------\n");
    printf("| %s\n",function);
    printf("+-----------------------------------------------------\n");
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

int PLC::getDISensorValue(SENSORS s) {


    TS7DataItem item[1];
    byte EB[1];

    switch(s) {
        case SENSORS::IR:
            item[0].Area     =S7AreaPE;
            item[0].WordLen  =S7WLByte;
            item[0].DBNumber =0;
            item[0].Start    =0;
            item[0].Amount   =1;
            item[0].pdata    =&EB;
            break;
        default:
            return -1;
    }

    int res=Client->ReadMultiVars(&item[0],1);

    if (Check(res,"Multiread Vars"))
    {
        return item[0].Result;
    } else {
        return -1;
    }
}