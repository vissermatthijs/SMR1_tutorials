//
// Created by ros-industrial on 9/26/17.
//

#ifndef SMR1_FRAME_H
#define SMR1_FRAME_H

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <string>

class Frame {
public:
    Frame(tf::TransformBroadcaster &br, std::string name, std::string parent, float x, float y, float z, float theta);

    void broadcast();
    void setParent(std::string p);

private:
    tf::TransformBroadcaster &br;
    tf::Transform transform;
    tf::Quaternion q;
    std::string name;
    std::string parent;

};

#endif //SMR1_FRAME_H
