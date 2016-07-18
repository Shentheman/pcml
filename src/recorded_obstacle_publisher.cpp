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
    const double future_time = 1.;
    const int future_frames = 2;

    ros::init(argc, argv, "recorded_obstacle_publisher");

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    ros::NodeHandle nh("~");

    // rate for
    ros::Rate rate(20);

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

    // skeleton data
    char filename[128];
    sprintf(filename, "/playpen/jaesungp/catkin_ws/src/pcml/data/s%s.txt", argv[1]);
    FILE* fp = fopen(filename, "r");

    int num_joints;
    fscanf(fp, "%d", &num_joints);

    std::vector<double> radius(num_joints);
    for (int i=0; i<num_joints; i++)
        fscanf(fp, "%lf", &radius[i]);

    int num_edges;
    fscanf(fp, "%d", &num_edges);

    std::vector<std::pair<int, int> > edge_indices(num_edges);
    for (int i=0; i<num_edges; i++)
        fscanf(fp, "%d %d", &edge_indices[i].first, &edge_indices[i].second);

    std::vector<std::vector<Eigen::Vector3d> > joint_positions;
    while (true)
    {
        Eigen::Vector3d p;
        if (fscanf(fp, "%lf %lf %lf", &p(0), &p(1), &p(2)) != 3)
            break;

        joint_positions.push_back(std::vector<Eigen::Vector3d>());
        for (int i=0; i<num_joints; i++)
        {
            if (i)
                fscanf(fp, "%lf %lf %lf", &p(0), &p(1), &p(2));

            joint_positions.back().push_back(p);
        }
    }

    bool first = true;
    int frame = 0;
    std::vector<Eigen::Vector3d> previous_joint_positions;
    while (ros::ok())
    {
        // joint positions at current frame
        const std::vector<Eigen::Vector3d>& new_joint_positions( joint_positions[frame] );

        // compute control inputs
        std::vector<Eigen::Vector3d> control_input(num_joints, Eigen::Vector3d::Zero());

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

        msg.header.frame_id = "camera_link";
        msg.header.stamp = ros::Time::now();

        pcml::FutureObstacleDistribution obstacle;

        for (int i=0; i<=future_frames; i++)
        {
            const double t = future_time * i / future_frames;

            // predict with zero control input
            std::vector<Eigen::Vector3d> control_input(num_joints, Eigen::Vector3d::Zero());
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

            for (int j=0; j<num_joints; j++)
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

        frame++;
        if (frame == joint_positions.size())
        {
            frame = 0;

            first = true;
            predictor.clear();
        }
    }

    return 0;
}
