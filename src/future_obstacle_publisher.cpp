#include <pcml/data/skeleton_stream.h>

// message
#include <pcml/FutureObstacleDistributions.h>
#include <std_msgs/String.h>

#include <ros/ros.h>


// simply generated distributions as same as input
static pcml::FutureObstacleDistributions getFutureObstacleDistributionsMessage(const std::vector<std::string>& joint_names, const Eigen::Matrix3Xd joints)
{
    pcml::FutureObstacleDistributions msg;
    pcml::FutureObstacleDistribution obstacle;

    msg.header.stamp = ros::Time::now();

    // not sure the frame id
    msg.header.frame_id = "camera_link";

    for (double t = 0; t <= 1; t += 1)
    {
        obstacle.future_time = t;
        for (int i=0; i<joint_names.size(); i++)
        {
            obstacle.obstacle_point.x = joints(0,i);
            obstacle.obstacle_point.y = joints(1,i);
            obstacle.obstacle_point.z = joints(2,i);

            for (int j=0; j<9; j++)
                obstacle.obstacle_covariance[j] = 0;
            obstacle.obstacle_covariance[0] = 0.05;
            obstacle.obstacle_covariance[4] = 0.05;
            obstacle.obstacle_covariance[8] = 0.05;

            obstacle.weight = 1.0;

            msg.obstacles.push_back(obstacle);
        }
    }

    return msg;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "future_obstacle_publisher");

    ros::NodeHandle nh("~");

    // rate for
    ros::Rate rate(30);

    // input stream
    pcml::SkeletonStream* stream;
    Eigen::Matrix3Xd joints;

    // publisher
    ros::Publisher future_obstacle_distributions_publisher;
    future_obstacle_distributions_publisher = nh.advertise<pcml::FutureObstacleDistributions>("future_obstacle_distributions", 1);

    while (future_obstacle_distributions_publisher.getNumSubscribers() < 1)
    {
        ROS_INFO("Waiting for [%s] subscriber...", future_obstacle_distributions_publisher.getTopic().c_str());
        fflush(stdout);
        ros::Duration(1.0).sleep();
    }
    ROS_INFO("Found subscriber");
    fflush(stdout);

    // parameters
    std::string input_stream_type;
    std::string joints_type;
    std::vector<std::string> joint_names;
    int human_model_divisor;
    bool render;

    // retrieve parameters
    nh.param<std::string>("input_stream_type", input_stream_type, "realtime");
    nh.param<std::string>("joints_type", joints_type, "upper_body");
    nh.param<int>("human_model_divisor", human_model_divisor, 4);
    nh.param<bool>("render", render, false);

    // joint names
    if (joints_type == "upper_body")
    {
        joint_names = pcml::SkeletonStream::jointNamesUpperBody();
    }
    else if (joints_type == "whole_body")
    {
        joint_names = pcml::SkeletonStream::jointNamesWholeBody();
    }
    else
    {
        ROS_ERROR("Undefined behavior for joints_type [%s]", joints_type.c_str());
        ROS_ERROR("Supported joints_type: upper_body, whole_body\n");
        fflush(stderr);
        return 1;
    }

    // input stream type
    if (input_stream_type == "realtime")
    {
        ROS_INFO("Streaming skeleton from Kinect");
        fflush(stdout);

        stream = new pcml::SkeletonRealtimeStream(nh, joint_names);
        stream->setVisualizationTopic(nh, "skeleton_realtime");
    }
    else if (input_stream_type == "cad120")
    {
        std::string cad120_directory;
        std::vector<int> cad120_input_info = {0, 0, 0};

        if (!nh.getParam("cad120_directory", cad120_directory))
        {
            ROS_ERROR("CAD120 directory is not specified in rosparam");
            fflush(stderr);
            return 1;
        }

        if (!nh.getParam("cad120_input_info", cad120_input_info))
        {
            ROS_WARN("CAD120 subject/action/video number is not specified, so using default values: 0/0/0");
            fflush(stdout);
        }

        ROS_INFO("Loading the specified CAD120 video from %s", cad120_directory.c_str());
        fflush(stdout);

        pcml::SkeletonCAD120Stream* cad120_stream = new pcml::SkeletonCAD120Stream(cad120_directory);
        cad120_stream->setVisualizationTopic(nh, "skeleton_cad120");
        cad120_stream->startReadFrames(cad120_input_info[0], cad120_input_info[1], cad120_input_info[2]);
        stream = cad120_stream;
    }
    else
    {
        ROS_ERROR("Undefined behavior for input_stream_type [%s]", input_stream_type.c_str());
        ROS_ERROR("Supported input_stream_type: realtime, cad120\n");
        fflush(stderr);
        return 1;
    }

    // publish future obstacles
    while (ros::ok())
    {
        // get skeleton joints
        if (stream->getSkeleton(joints))
        {
            // render
            if (render)
            {
                stream->renderSkeleton(joints);
            }

            // TODO: compute future obstacles
            pcml::FutureObstacleDistributions future_obstacle_distributions;

            // TODO: compute future obstacles. The current code publishes the same human model as the input.
            future_obstacle_distributions = getFutureObstacleDistributionsMessage(joint_names, joints);

            // publish
            future_obstacle_distributions_publisher.publish(future_obstacle_distributions);
        }
        else
        {
            ROS_WARN("Failed to retrieve skeleton from stream");
            fflush(stdout);
        }

        rate.sleep();
    }

    delete stream;

    ros::shutdown();
    return 0;
}
