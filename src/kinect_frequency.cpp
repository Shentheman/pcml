#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <stdio.h>

void colorImageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    static double last_time = 0;
    const double current_time = ros::Time::now().toSec();
    if (last_time != 0)
    {
        printf("color (%dx%d) time: %lf\n",
               msg->width,
               msg->height,
               (current_time - last_time) * 1000.0);
        fflush(stdout);
    }
    last_time = current_time;
}

void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    static double last_time = 0;
    const double current_time = ros::Time::now().toSec();
    if (last_time != 0)
    {
        printf("depth (%dx%d) time: %lf\n",
               msg->width,
               msg->height,
               (current_time - last_time) * 1000.0);
        fflush(stdout);
    }
    last_time = current_time;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kienct_frequency");

    ros::NodeHandle nh;
    ros::Subscriber subscriber_color = nh.subscribe("/camera/rgb/image_color", 1000, colorImageCallback);
    ros::Subscriber subscriber_depth = nh.subscribe("/camera/depth_registered/image", 1000, depthImageCallback);

    ros::spin();
    return 0;
}
