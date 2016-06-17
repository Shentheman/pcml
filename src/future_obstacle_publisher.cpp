#include <pcml/data/skeleton_stream.h>

// message
#include <pcml/FutureObstacleDistributions.h>
#include <std_msgs/String.h>

#include <ros/ros.h>


std::vector<std::pair<std::string, double> > radii =
{
    {"head"          , 0.10},
    {"neck"          , 0.05},
    {"torso"         , 0.15},
    {"left_shoulder" , 0.05},
    {"left_elbow"    , 0.05},
    {"left_hand"     , 0.05},
    {"right_shoulder", 0.05},
    {"right_elbow"   , 0.05},
    {"right_hand"    , 0.05},
};

std::vector<std::pair<const char*, const char*> > edges =
{
    {"head", "neck"},
    {"neck", "torso"},
    {"neck", "left_shoulder"},
    {"left_shoulder", "left_elbow"},
    {"left_elbow", "left_hand"},
    {"neck", "right_shoulder"},
    {"right_shoulder", "right_elbow"},
    {"right_elbow", "right_hand"},
};

// simply generated distributions as same as input
static pcml::FutureObstacleDistributions getFutureObstacleDistributionsMessage(const std::vector<std::string>& joint_names, const Eigen::Matrix3Xd joints)
{
    const int num_samples = 2;

    pcml::FutureObstacleDistributions msg;
    pcml::FutureObstacleDistribution obstacle;

    msg.header.stamp = ros::Time::now();

    // not sure the frame id
    msg.header.frame_id = "camera_depth_frame";

    for (int j=0; j<9; j++)
        obstacle.obstacle_covariance[j] = 0;
    obstacle.obstacle_covariance[0] = 0.05;
    obstacle.obstacle_covariance[4] = 0.05;
    obstacle.obstacle_covariance[8] = 0.05;

    for (double t = 0; t <= 1; t += 1)
    {
        obstacle.future_time = t;
        for (int i=0; i<edges.size(); i++)
        {
            int joint0;
            int joint1;
            for (int j=0; j<joint_names.size(); j++)
            {
                if (joint_names[j] == edges[i].first)
                    joint0 = j;
                if (joint_names[j] == edges[i].second)
                    joint1 = j;
            }

            double r0;
            double r1;
            for (int j=0; j<radii.size(); j++)
            {
                if (joint_names[joint0] == radii[j].first)
                    r0 = radii[j].second;
                if (joint_names[joint1] == radii[j].first)
                    r1 = radii[j].second;
            }

            const Eigen::Vector3d p0 = joints.col(joint0);
            const Eigen::Vector3d p1 = joints.col(joint1);
            for (int j=0; j<num_samples; j++)
            {
                const double u = (j+0.5) / num_samples;
                const double radius = (1.-u) * r0 + u * r1;
                const Eigen::Vector3d position = (1.-u) * p0 + u * p1;

                obstacle.obstacle_point.x = position(0);
                obstacle.obstacle_point.y = position(1);
                obstacle.obstacle_point.z = position(2);

                obstacle.weight = 1.0;
                obstacle.radius = radius;

                msg.obstacles.push_back(obstacle);
            }
        }
    }

    return msg;
}


int main(int argc, char** argv)
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    
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
        ros::Duration(1.0).sleep();
    }
    ROS_INFO("Found subscriber");

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
    nh.param<bool>("render", render, true);

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
        return 1;
    }

    // input stream type
    if (input_stream_type == "realtime")
    {
        ROS_INFO("Streaming skeleton from Kinect");

        stream = new pcml::SkeletonRealtimeStream(nh, joint_names);
        stream->setVisualizationTopic(nh, "skeleton_realtime");
        
        ROS_INFO("waiting 1 sec for publisher/subscriber setup");
        ros::Duration(1.0).sleep();
    }
    else if (input_stream_type == "cad120")
    {
        std::string cad120_directory;
        std::vector<int> cad120_input_info = {0, 0, 0};

        if (!nh.getParam("cad120_directory", cad120_directory))
        {
            ROS_ERROR("CAD120 directory is not specified in rosparam");
            return 1;
        }

        if (!nh.getParam("cad120_input_info", cad120_input_info))
        {
            ROS_WARN("CAD120 subject/action/video number is not specified, so using default values: 0/0/0");
        }

        ROS_INFO("Loading the specified CAD120 video from %s", cad120_directory.c_str());

        pcml::SkeletonCAD120Stream* cad120_stream = new pcml::SkeletonCAD120Stream(cad120_directory);
        cad120_stream->setVisualizationTopic(nh, "skeleton_cad120");
        cad120_stream->startReadFrames(cad120_input_info[0], cad120_input_info[1], cad120_input_info[2]);
        stream = cad120_stream;
    }
    else
    {
        ROS_ERROR("Undefined behavior for input_stream_type [%s]", input_stream_type.c_str());
        ROS_ERROR("Supported input_stream_type: realtime, cad120\n");
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
                stream->renderSkeleton(joints);

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
        }

        rate.sleep();
    }

    delete stream;

    ros::shutdown();
    return 0;
}
