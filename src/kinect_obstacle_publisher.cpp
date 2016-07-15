#include <pcml/data/kinect_skeleton_stream.h>
#include <pcml/kalman_filter/points_kalman_filter.h>

// message
#include <pcml/FutureObstacleDistributions.h>
#include <std_msgs/String.h>
#include <eigen_conversions/eigen_msg.h>

#include <ros/ros.h>


int main(int argc, char** argv)
{
    const double length_max_variation = 1.1;
    const double future_time = 3.;
    const int future_frames = 10;

    ros::init(argc, argv, "kinect2_obstacle_publisher");

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    ros::NodeHandle nh("~");

    // rate for
    ros::Rate rate(15);

    std::vector<std::string> joint_names;
    nh.getParam("joint_names", joint_names);

    std::vector<std::string> edges;
    nh.getParam("edges", edges);

    std::map<std::string, double> radius_map;
    nh.getParam("radius", radius_map);

    std::map<std::string, int> joint_indices;
    for (int i=0; i<joint_names.size(); i++)
        joint_indices[ joint_names[i] ] = i;

    std::vector<std::pair<int, int> > edge_indices;
    for (int i=0; i<edges.size(); i+=2)
        edge_indices.push_back(std::make_pair( joint_indices[edges[i]], joint_indices[edges[i+1]] ));

    std::vector<double> radius(joint_names.size(), 0.05);
    for (std::map<std::string, double>::const_iterator it = radius_map.cbegin(); it != radius_map.cend(); it++)
        radius[joint_indices[it->first]] = it->second;

    // input stream
    pcml::KinectSkeletonStream stream;
    stream.setSkeleton(joint_names, edge_indices);

    // predictor
    pcml::PointsKalmanFilterPredictor predictor;
    predictor.setDeltaT(rate.expectedCycleTime().toSec());
    predictor.setDiagonalMeasurementNoise(0.0001);
    predictor.setDiagonalControlNoise(0.0001);

    // publisher
    // TODO: topic namespace
    ros::Publisher future_obstacle_distributions_publisher;
    future_obstacle_distributions_publisher = nh.advertise<pcml::FutureObstacleDistributions>("/future_obstacle_publisher/future_obstacle_distributions", 1);

    // wait for ros setup
    ros::Duration(0.5).sleep();

    bool first = true;
    std::vector<Eigen::Vector3d> previous_joint_positions;
    while (ros::ok())
    {
        // joint positions at current frame
        stream.read();
        const std::vector<Eigen::Vector3d>& new_joint_positions = stream.getJointPositions();

        // compute control inputs
        std::vector<Eigen::Vector3d> control_input(joint_names.size(), Eigen::Vector3d::Zero());

        std::vector<double> lengths;

        if (!first)
        {
            // edge lengths of previous frame
            for (int i=0; i<edge_indices.size(); i++)
                lengths.push_back( (previous_joint_positions[edge_indices[i].first] - previous_joint_positions[edge_indices[i].second]).norm() );

            // predicted joint positions without observation
            std::vector<Eigen::Matrix<double, 6, 1> > mu;
            std::vector<Eigen::Matrix<double, 6, 6> > sigma;
            predictor.predict(rate.expectedCycleTime().toSec(), control_input, mu, sigma);

            // edge length of current frame
            for (int i=0; i<edge_indices.size(); i++)
            {
                const int i0 = edge_indices[i].first;
                const int i1 = edge_indices[i].second;

                const double length = (mu[i0].block(0, 0, 3, 1) - mu[i1].block(0, 0, 3, 1)).norm();

                if (length > lengths[i] * length_max_variation)
                {
                    double over = length - lengths[i] * length_max_variation;
                    control_input[i1] += (mu[i0].block(0, 0, 3, 1) - mu[i1].block(0, 0, 3, 1)).normalized() * over;
                }
            }
        }

        predictor.iterate(new_joint_positions, control_input);

        // publish future obstacles
        std::vector<Eigen::Matrix<double, 6, 1> > mu;
        std::vector<Eigen::Matrix<double, 6, 6> > sigma;
        pcml::FutureObstacleDistributions msg;

        msg.header.frame_id = "map";
        msg.header.stamp = ros::Time::now();

        pcml::FutureObstacleDistribution obstacle;

        for (int i=0; i<=future_frames; i++)
        {
            const double t = future_time * i / future_frames;

            // predict with zero control input
            std::vector<Eigen::Vector3d> control_input(joint_names.size(), Eigen::Vector3d::Zero());
            predictor.predict(t, control_input, mu, sigma);

            if (!first)
            {
                // edge length of current frame
                for (int j=0; j<edge_indices.size(); j++)
                {
                    const int j0 = edge_indices[j].first;
                    const int j1 = edge_indices[j].second;

                    Eigen::Vector3d v = (mu[j0].block(0, 0, 3, 1) + control_input[j0] - mu[j1].block(0, 0, 3, 1));
                    const double length = v.norm();

                    if (length > lengths[j] * length_max_variation)
                    {
                        double over = length - lengths[j] * length_max_variation;
                        control_input[j1] += v.normalized() * over;
                    }
                }

                // predict with computed control inputs
                predictor.predict(t, control_input, mu, sigma);
            }

            for (int j=0; j<joint_names.size(); j++)
            {
                obstacle.future_time = t;
                obstacle.weight = 0.1;
                tf::pointEigenToMsg(mu[j].block(0, 0, 3, 1), obstacle.obstacle_point);
                for (int k=0; k<9; k++)
                    obstacle.obstacle_covariance[k] = sigma[j](k/3, k%3);
                obstacle.radius = radius[j];

                msg.obstacles.push_back(obstacle);
            }
        }

        future_obstacle_distributions_publisher.publish(msg);

        previous_joint_positions = new_joint_positions;
        first = false;

        rate.sleep();
    }

    return 0;
}
