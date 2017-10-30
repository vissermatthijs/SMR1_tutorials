//
// Created by ros-industrial on 9/13/17.
//

#include <robot/MoveActionServer.h>
#include <string>
#include <sstream>
#include <iostream>

MoveActionServer::MoveActionServer(ros::NodeHandle &n) : planner(n, false),
                                                         name("MovePlant"),
                                                         as_(n, name,
                                                             boost::bind(&MoveActionServer::exec, this, _1),
                                                             false) {

    this->as_.start();


    counter[0].first = 1;
    counter[0].second = 0;

    counter[1].first = 0;
    counter[1].second = 0;

    counter[2].first = -1;
    counter[2].second = 0;

    counter[3].first = 0;
    counter[3].second = 1;

    this->middleCounter = 0;
    this->distance = 0.115f;
}


void MoveActionServer::exec(const robot::MovePlantGoalConstPtr &goal) {

    std::cout << "Goal type: " <<goal->type <<std::endl;

    if(goal->type == 0) {
        this->planner.manualPose("home");
    } else if(goal->type >= 1) {

        this->planner.manualPose("pickup_step1");
        this->planner.manualPose("pickup_step2");
        this->planner.manualPose("pickup_step3");

        //this->planner.pushCollisionObject();
        //this->planner.pushConstraintFromCurrentOrientation();

        if(goal->type == 1) {
            this->planner.manualPose("place_bin1");
            geometry_msgs::Pose p = this->planner.getCurrentPose();
            std::cout << "Counter first" << counter[1].first << std::endl;
            p.position.y += counter[1].first * this->distance;
            p.position.x += counter[1].second * this->distance;
            this->planner.manualPose(p);

            p.position.z -= 0.30f;
            this->planner.manualPose(p);
            p.position.z -= 0.10f;
            this->planner.manualPose(p);

            p.position.x += 0.15f;
            p.position.y += 0.15f;
            this->planner.manualPose(p);

            p.position.z += 0.40f;
            this->planner.manualPose(p);

            counter[1].first++;

            if(counter[1].first > 4) {
                counter[1].first = 0;
                counter[1].second = 1;
            }

        } else if(goal->type == 2) {
            geometry_msgs::Pose p = this->planner.getCurrentPose();

            if(this->middleCounter > 9) {
                this->middleCounter = 0;
            }

            if(this->middleCounter == 8) {
                this->planner.manualPose("place_bin2_2");
                p = this->planner.getCurrentPose();

                p.position.z -= 0.30f;
                this->planner.manualPose(p);
                p.position.z -= 0.10f;
                this->planner.manualPose(p);

                p.position.y += 0.15f;
                this->planner.manualPose(p);

                p.position.z += 0.40f;
                this->planner.manualPose(p);

            } else if(this->middleCounter == 9) {
                this->planner.manualPose("place_bin2_2");
                p = this->planner.getCurrentPose();

                p.position.z -= 0.30f;
                this->planner.manualPose(p);
                p.position.z -= 0.10f;
                this->planner.manualPose(p);

                p.position.y += 0.15f;
                this->planner.manualPose(p);

                p.position.z += 0.40f;
                this->planner.manualPose(p);

            } else if(this->middleCounter % 2 == 0) {
                // voorste row
                this->planner.manualPose("place_bin2_1");
                p = this->planner.getCurrentPose();

                counter[2].first++;
                counter[2].second = 0;
                p.position.y -= counter[2].first * distance;
                p.position.x -= counter[2].second * distance;
                this->planner.manualPose(p);

                p.position.z -= 0.30f;
                this->planner.manualPose(p);
                p.position.z -= 0.0f;
                this->planner.manualPose(p);

                p.position.x -= 0.15f;
                this->planner.manualPose(p);

                p.position.z += 0.40f;
                this->planner.manualPose(p);
            } else {
                // achterste row
                this->planner.manualPose("place_bin2_1");
                p = this->planner.getCurrentPose();

                counter[2].second = 1;
                p.position.y -= counter[2].first * distance;
                p.position.x -= counter[2].second * distance;
                this->planner.manualPose(p);

                p.position.z -= 0.30f;
                this->planner.manualPose(p);
                p.position.z -= 0.0f;
                this->planner.manualPose(p);

                //this->planner.pushConstraintFromCurrentOrientation();
                p.position.x -= 0.15f;
                this->planner.manualPose(p);;

                p.position.z += 0.40f;
                this->planner.manualPose(p);
            }

            this->middleCounter++;

        } else if(goal->type >= 3) {
            this->planner.manualPose("place_bin3");
            geometry_msgs::Pose p = this->planner.getCurrentPose();

            std::cout << "PRE PLACE BIN 3 X:" << counter[3].first << " Y:" << counter[3].second << std::endl;
            p.position.y += counter[3].first * this->distance;
            p.position.x -= counter[3].second * this->distance;
            //Move to empty space in tray
            this->planner.manualPose(p);

            std::cout << "LOWERR PLANT" <<std::endl;
            //lower plant
            p.position.z -= 0.20f;
            this->planner.manualPose(p);
            p.position.z -= 0.10f;
            this->planner.manualPose(p);

            //remove eoat from plant
            p.position.x -= 0.15f;
            p.position.y += 0.15f;
            this->planner.manualPose(p);

            p.position.z += 0.40f;
            this->planner.manualPose(p);

            counter[3].first++;

            if(counter[3].first > 4) {
                counter[3].first = 0;
                counter[3].second = 1;
            }
        }

        this->planner.manualPose("home");
    }

    this->result.sequence = true;
    this->as_.setSucceeded(this->result);
}

void MoveActionServer::test() {
  //  this->planner.manualPose(-0.214f, -0.245f, 1.84f, -1.0f);

  /*  std::stringstream ss;
    ss << "[Position] X: " << this->planner.getCurrentPose().pose.position.x << " Y: " <<
       this->planner.getCurrentPose().pose.position.y << " Z: " << this->planner.getCurrentPose().pose.position.z;

    ROS_INFO(ss.str().c_str());

    std::stringstream ss2;
    ss2 << "[Orientation] X: " << this->planner.getCurrentPose().pose.orientation.x << " Y: " <<
       this->planner.getCurrentPose().pose.orientation.y << " Z: " << this->planner.getCurrentPose().pose.orientation.z <<
        " W: " << this->planner.getCurrentPose().pose.orientation.w;

    ROS_INFO(ss2.str().c_str()); */
}

