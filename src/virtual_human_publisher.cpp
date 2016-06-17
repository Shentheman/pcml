#include <pcml/data/skeleton_stream.h>

// message
#include <pcml/FutureObstacleDistributions.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>

#include <ros/ros.h>


std::vector<std::pair<std::string, double> > radii =
{
    {"head"          , 0.15},
    {"neck"          , 0.10},
    {"torso"         , 0.20},
    {"left_shoulder" , 0.10},
    {"left_elbow"    , 0.10},
    {"left_hand"     , 0.10},
    {"right_shoulder", 0.10},
    {"right_elbow"   , 0.10},
    {"right_hand"    , 0.10},
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


bool requested;
double delay;
double speed;

void requestCallback(const std_msgs::Float64MultiArrayConstPtr& msg)
{
    requested = true;
    delay = msg->data[0];
    speed = msg->data[1];
}


int main(int argc, char** argv)
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    
    ros::init(argc, argv, "future_obstacle_publisher");

    ros::NodeHandle nh("future_obstacle_publisher");

    // rate for
    ros::Rate rate(30);

    // publisher
    ros::Publisher future_obstacle_distributions_publisher;
    future_obstacle_distributions_publisher = nh.advertise<pcml::FutureObstacleDistributions>("future_obstacle_distributions", 1);

    // subscriber
    ros::Subscriber request_subscriber;
    request_subscriber = nh.subscribe<std_msgs::Float64MultiArray>("virtual_human_arm_request", 1, &requestCallback);

    // joint information
    std::vector<std::string> joint_names;
    Eigen::Matrix3Xd default_joints;
    for (int i=0; i<radii.size(); i++)
        joint_names.push_back(radii[i].first);

    default_joints.resize(Eigen::NoChange, joint_names.size());
    default_joints.col(0) = Eigen::Vector3d( 1.00, 0.00, 0.23);
    default_joints.col(1) = Eigen::Vector3d( 1.00, 0.00, 0.00);
    default_joints.col(2) = Eigen::Vector3d( 1.00, 0.00,-0.50);
    default_joints.col(3) = Eigen::Vector3d( 1.00, 0.20, 0.00);
    default_joints.col(4) = Eigen::Vector3d( 1.00, 0.20,-0.30);
    default_joints.col(5) = Eigen::Vector3d( 1.00, 0.20,-0.60);
    default_joints.col(6) = Eigen::Vector3d( 1.00,-0.20, 0.00);
    default_joints.col(7) = Eigen::Vector3d( 1.00,-0.20,-0.30);
    default_joints.col(8) = Eigen::Vector3d( 1.00,-0.20,-0.60);

    ros::WallTime start_time;

    requested = true;
    delay = 1.0;
    speed = 1.0;

    while (ros::ok())
    {
        if (requested)
        {
            ROS_INFO("virtual arm movement requested, delay: %lf, speed: %lf", delay, speed);
            start_time = ros::WallTime::now();
            requested = false;
        }

        Eigen::Matrix3Xd joints;
        joints = default_joints;

        const double t = (ros::WallTime::now() - start_time).toSec();
        if (delay <= t && t <= 5.0)
        {
            const double angle = std::min(speed * (t - delay), M_PI / 2.0);

            Eigen::Affine3d transform;
            transform.setIdentity();
            transform.translate(default_joints.col(3)).rotate(Eigen::Quaterniond(std::cos(angle/2.), 0., std::sin(angle/2.), 0.)).translate(-default_joints.col(3));
            joints.col(4) = transform * default_joints.col(4);
            joints.col(5) = transform * default_joints.col(5);
        }

        pcml::FutureObstacleDistributions msg = getFutureObstacleDistributionsMessage(joint_names, joints);
        future_obstacle_distributions_publisher.publish(msg);
        rate.sleep();

        ros::spinOnce();
    }

    ros::shutdown();
    return 0;
}
