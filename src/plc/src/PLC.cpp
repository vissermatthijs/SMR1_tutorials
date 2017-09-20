//
// Created by ros-industrial on 9/20/17.
//

#include <plc/PLC.h>
#include <iostream>
#include <ros/ros.h>

PLC::PLC(std::string ip) : rack(0), slot(2) {

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