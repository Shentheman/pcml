#ifndef PCML_KINECT_SKELETON_STREAM_H
#define PCML_KINECT_SKELETON_STREAM_H


#include <vector>
#include <string>

#include <Eigen/Dense>

#include <tf/transform_listener.h>


namespace pcml
{

class KinectSkeletonStream
{
public:

    KinectSkeletonStream();

    inline const std::vector<Eigen::Vector3d>& getJointPositions() const
    {
        return joint_positions_;
    }

    void setSkeleton(const std::vector<std::string>& joint_names, const std::vector<std::pair<int, int> > edges);

    void read();

private:

    std::vector<std::string> joint_names_;
    std::vector<std::pair<int, int> > edges_;

    std::vector<Eigen::Vector3d> joint_positions_;

    tf::TransformListener transform_listener_;
};

}


#endif // PCML_KINECT_SKELETON_STREAM_H
