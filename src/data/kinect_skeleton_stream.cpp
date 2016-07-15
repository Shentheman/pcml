#include <pcml/data/kinect_skeleton_stream.h>


namespace pcml
{

KinectSkeletonStream::KinectSkeletonStream()
{
}

void KinectSkeletonStream::setSkeleton(const std::vector<std::string>& joint_names, const std::vector<std::pair<int, int> > edges)
{
    joint_names_ = joint_names;
    edges_ = edges;

    joint_positions_ = std::vector<Eigen::Vector3d>(joint_names_.size(), Eigen::Vector3d(0., 0., 0.));
}

void KinectSkeletonStream::read()
{
    for (int i=0; i<joint_names_.size(); i++)
    {
        std::string error_string;
        ros::Time time;
        transform_listener_.getLatestCommonTime("map", joint_names_[i], time, &error_string);

        tf::StampedTransform transform;
        try
        {
            transform_listener_.lookupTransform("kinect_depth_frame", joint_names_[i], time, transform);

            tf::Vector3 position = transform.getOrigin();
            joint_positions_[i] = Eigen::Vector3d(position.x(), position.y(), position.z());
        }
        catch (const tf::TransformException& ex)
        {
            ROS_ERROR("%s", ex.what());
        }
    }
}

}
