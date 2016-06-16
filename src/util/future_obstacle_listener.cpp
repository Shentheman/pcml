#include <pcml/util/future_obstacle_listener.h>


namespace pcml
{

void FutureObstacleListener::futureObstacleCallbackFuction(const pcml::FutureObstacleDistributionsConstPtr& msg)
{
    boost::mutex::scoped_lock(future_obstacle_mutex_);
    future_obstacle_distributions_ = *msg;
}

FutureObstacleListener::FutureObstacleListener()
    : spinner_(1, &callback_queue_)
{
    ros::NodeHandle node_handle;

    ros::SubscribeOptions options;
    options.template init<pcml::FutureObstacleDistributions>("/future_obstacle_publisher/future_obstacle_distributions", 1, std::bind(&FutureObstacleListener::futureObstacleCallbackFuction, this, std::placeholders::_1));
    options.callback_queue = &callback_queue_;

    future_obstacle_distribution_subscriber_ = node_handle.subscribe(options);

    spinner_.start();
}

FutureObstacleListener::~FutureObstacleListener()
{
}

std::string FutureObstacleListener::getFrameId()
{
    boost::mutex::scoped_lock(future_obstacle_mutex_);
    return future_obstacle_distributions_.header.frame_id;
}

pcml::FutureObstacleDistributions FutureObstacleListener::getObstaclesAtCurrentTime()
{
    pcml::FutureObstacleDistributions current_obstacles;

    {
        boost::mutex::scoped_lock(future_obstacle_mutex_);

        current_obstacles.header = future_obstacle_distributions_.header;

        for (int i=0; i<future_obstacle_distributions_.obstacles.size(); i++)
        {
            pcml::FutureObstacleDistribution distribution = future_obstacle_distributions_.obstacles[i];

            if (distribution.future_time == 0.)
            {
                distribution.obstacle_covariance.fill(0.);
                current_obstacles.obstacles.push_back( distribution );
            }
        }
    }

    return current_obstacles;
}

pcml::FutureObstacleDistributions FutureObstacleListener::getFutureObstacleDistributions()
{
    boost::mutex::scoped_lock(future_obstacle_mutex_);
    return future_obstacle_distributions_;
}

}
