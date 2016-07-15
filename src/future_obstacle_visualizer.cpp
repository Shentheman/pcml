#include <pcml/data/skeleton_stream.h>

// message
#include <pcml/FutureObstacleDistributions.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/String.h>

#include <ros/ros.h>


// subscriber
static bool callback_called = false;
static pcml::FutureObstacleDistributions future_obstacle_distributions;
static void futureObstacleDistributionsCallback(const pcml::FutureObstacleDistributions::ConstPtr& msg)
{
    callback_called = true;
    future_obstacle_distributions = *msg;
}

// visualization marker generator
void generateEllipsoidMarker(double radius, const Eigen::Vector3d& position, const Eigen::Matrix3d& covariance, visualization_msgs::Marker& marker)
{
    marker.type = visualization_msgs::Marker::SPHERE;

    marker.pose.position.x = position(0);
    marker.pose.position.y = position(1);
    marker.pose.position.z = position(2);

    // axis: eigenvectors
    // radius: eigenvalues
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& r = svd.singularValues();
    Eigen::Matrix3d Q = svd.matrixU();

    // to make determinant 1
    if (Q.determinant() < 0)
        Q.col(2) *= -1.;
    const Eigen::Quaterniond q(Q);

    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();

    /*
    chi_square_3d_table_[0.95 ] =  0.35;
    chi_square_3d_table_[0.90 ] =  0.58;
    chi_square_3d_table_[0.80 ] =  1.01;
    chi_square_3d_table_[0.70 ] =  1.42;
    chi_square_3d_table_[0.50 ] =  2.37;
    chi_square_3d_table_[0.30 ] =  3.66;
    chi_square_3d_table_[0.20 ] =  4.64;
    chi_square_3d_table_[0.10 ] =  6.25;
    chi_square_3d_table_[0.05 ] =  7.82;
    chi_square_3d_table_[0.01 ] = 11.34;
    chi_square_3d_table_[0.001] = 16.27;
    */

    marker.scale.x = 2. * (7.82 * std::sqrt(r(0)) + radius);
    marker.scale.y = 2. * (7.82 * std::sqrt(r(1)) + radius);
    marker.scale.z = 2. * (7.82 * std::sqrt(r(2)) + radius);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "future_obstacle_visualizer");

    ros::NodeHandle nh;

    // rate for visualization msg
    ros::Rate rate(30);

    // subscriber
    ros::Subscriber future_obstacle_distributions_subscriber;
    future_obstacle_distributions_subscriber = nh.subscribe("/future_obstacle_publisher/future_obstacle_distributions", 1, futureObstacleDistributionsCallback);

    // visualization msg publisher
    ros::Publisher visualization_publisher;
    visualization_publisher = nh.advertise<visualization_msgs::MarkerArray>("future_obstacle_distributions_marker_array", 1);

    while (future_obstacle_distributions_subscriber.getNumPublishers() < 1)
    {
        ROS_INFO("Waiting for [%s] publisher...", future_obstacle_distributions_subscriber.getTopic().c_str());
        fflush(stdout);
        ros::Duration(1.0).sleep();
    }
    ROS_INFO("Found publisher");
    fflush(stdout);

    // wait until the callback is called
    while (ros::ok() && !callback_called)
    {
        ros::spinOnce();
        rate.sleep();
    }

    ROS_INFO("Future obstacle distributions subscriber setup");
    fflush(stdout);

    // visualize
    while (ros::ok())
    {
        // visualize
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker;

        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = future_obstacle_distributions.header.frame_id;   // not sure

        marker.action = visualization_msgs::Marker::ADD;

        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        marker.ns = "future_obstacle_distrtibutions";

        for (int i=0; i<future_obstacle_distributions.obstacles.size(); i++)
        {
            marker.id = i;

            const pcml::FutureObstacleDistribution& obstacle = future_obstacle_distributions.obstacles[i];

            const double& radius = obstacle.radius;
            const Eigen::Vector3d position( obstacle.obstacle_point.x, obstacle.obstacle_point.y, obstacle.obstacle_point.z );
            const Eigen::Matrix3d covariance = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> >(&obstacle.obstacle_covariance[0]);

            generateEllipsoidMarker(radius, position, covariance, marker);

            marker.color.a = obstacle.weight;

            marker_array.markers.push_back( marker );
        }
        visualization_publisher.publish(marker_array);

        // retrieve the next message in queue
        ros::spinOnce();

        rate.sleep();
    }

    ros::shutdown();
    return 0;
}
