//
// Created by ros-industrial on 9/26/17.
//

#include <item_frame/Frame.h>

Frame::Frame(tf::TransformBroadcaster &br, std::string name, float x, float y, float z, float theta) : br(br),
name(name), parent("world") {

    this->transform.setOrigin( tf::Vector3(x, y, z));
    this->q.setRPY(0, 0, theta);
    this->transform.setRotation(q);
}

void Frame::broadcast() {
    this->br.sendTransform(tf::StampedTransform(this->transform, ros::Time::now(), this->parent, this->name));
}

void Frame::setParent(std::string p) {
    this->parent = p;
}

