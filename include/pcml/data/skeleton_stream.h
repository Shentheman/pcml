#ifndef PCML_SKELETON_STREAM_H
#define PCML_SKELETON_STREAM_H


#include <pcml/data/cad120_reader.h>

#include <eigen3/Eigen/Dense>

#include <ros/ros.h>
#include <tf/transform_listener.h>


namespace pcml
{

class SkeletonStream
{
public:

    static std::vector<std::string> jointNamesWholeBody();
    static std::vector<std::string> jointNamesUpperBody();

public:

    SkeletonStream();
    SkeletonStream(const std::vector<std::string>& joint_names);

    void setVisualizationTopic(ros::NodeHandle nh, const std::string& topic);

    virtual bool getSkeleton(Eigen::Matrix3Xd& skeleton);

    void renderSkeleton();

    inline bool isFinished()
    {
        return finished_;
    }

protected:

    inline virtual std::string getFrameId()
    {
        return "/camera_link";
    }

    bool finished_;
    std::vector<std::string> joint_names_;

private:

    ros::Publisher marker_array_publisher_;
};


class SkeletonFileStream : public SkeletonStream
{
public:

    /**
     *  Get skeleton from input file.
     *  Can be real-time or frame-by-frame.
     */
    SkeletonFileStream();
    SkeletonFileStream(const std::vector<std::string>& joint_names);

    virtual bool getSkeleton(Eigen::Matrix3Xd& skeleton);

private:

    bool is_realtime_;
};


class SkeletonCAD120Stream : public SkeletonFileStream
{
private:

    /// conversion to CAD120 joint names
    static const std::map<std::string, std::string> joint_name_conversion_map_;

public:

    /**
     *  Get skeleton from CAD120 input file.
     *  Can be real-time or frame-by-frame.
     */
    SkeletonCAD120Stream(const std::string& directory);
    SkeletonCAD120Stream(const std::string& directory, const std::vector<std::string>& joint_names);

    // CAD120 wrapper functions
    int numSubjects();
    int numActions(int subject);
    int numVideos(int subject, int action);
    void startReadFrames(int subject, int action, int video);

    virtual bool getSkeleton(Eigen::Matrix3Xd& skeleton);

private:

    CAD120Reader reader_;
};


class SkeletonRealtimeStream : public SkeletonStream
{
public:

    static const int MAX_USER_ID = 10;

public:

    /**
     *  Get skeleton from TF published by openni_tracker running real-time.
     */
    SkeletonRealtimeStream(const ros::NodeHandle& nh);
    SkeletonRealtimeStream(const ros::NodeHandle& nh, const std::vector<std::string>& joint_names);

    virtual bool getSkeleton(Eigen::Matrix3Xd& skeleton);

private:

    inline virtual std::string getFrameId()
    {
        return nh_.getNamespace() + "camera_depth_frame";
    }

    int getUserId();

    const ros::NodeHandle& nh_;
    tf::TransformListener listener_;
};

}


#endif // PCML_SKELETON_STREAM_H
