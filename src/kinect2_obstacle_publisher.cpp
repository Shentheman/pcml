#include <pcml/data/kinect2_skeleton_stream.h>
#include <pcml/kalman_filter/points_kalman_filter.h>

// message
#include <pcml/FutureObstacleDistributions.h>
#include <std_msgs/String.h>

#include <ros/ros.h>


int main(int argc, char** argv)
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    
    ros::init(argc, argv, "kinect2_obstacle_publisher");

    ros::NodeHandle nh("~");

    // rate for
    ros::Rate rate(30);

    // input stream
    pcml::Kinect2SkeletonStream stream;

    // predictor
    pcml::PointsKalmanFilterPredictor predictor;

    // publisher
    ros::Publisher future_obstacle_distributions_publisher;
    future_obstacle_distributions_publisher = nh.advertise<pcml::FutureObstacleDistributions>("future_obstacle_distributions", 1);

    while (ros::ok())
    {
        ROS_ERROR("KINECT2 OBSTACLE PUBLISHER NOT IMPLEMENTED");

        rate.sleep();
    }

    return 0;
}
