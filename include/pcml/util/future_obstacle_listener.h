#ifndef PCML_FUTURE_OBSTACLE_LISTENER_H
#define PCML_FUTURE_OBSTACLE_LISTENER_H


#include <ros/ros.h>
#include <ros/callback_queue.h>

#include <pcml/FutureObstacleDistributions.h>


namespace pcml
{

class FutureObstacleListener
{
public:

    FutureObstacleListener();
    ~FutureObstacleListener();

    pcml::FutureObstacleDistributions getObstaclesAtCurrentTime();
    pcml::FutureObstacleDistributions getFutureObstacleDistributions();
    std::string getFrameId();

private:

    void futureObstacleCallbackFuction(const pcml::FutureObstacleDistributionsConstPtr& msg);

    ros::Subscriber future_obstacle_distribution_subscriber_;
    ros::AsyncSpinner spinner_;
    ros::CallbackQueue callback_queue_;

    boost::mutex future_obstacle_mutex_;
    pcml::FutureObstacleDistributions future_obstacle_distributions_;
};

}


#endif // PCML_FUTURE_OBSTACLE_LISTENER_H
