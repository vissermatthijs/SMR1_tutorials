//
// Created by ros-industrial on 9/20/17.
//

#ifndef SMR1_PLC_H
#define SMR1_PLC_H

#include <plc/snap7.h>

class PLC {
public:
    PLC(std::string ip);
    TS7CpuInfo getCPUInfo() const;
    TS7CpInfo getCPInfo() const;

    ~PLC();

private:

    TS7Client *Client;
    int rack,slot;

    bool Check(int Result, const char * function);


};

#endif //SMR1_PLC_H
