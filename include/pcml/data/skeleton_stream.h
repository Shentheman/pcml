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

    /// get the skeleton at the crurent time and then render skeleton
    void renderSkeleton();

    /// render skeleton
    void renderSkeleton(const Eigen::Matrix3Xd& skeleton);

    inline bool isFinished()
    {
        return finished_;
    }

protected:

    inline virtual std::string getFrameId()
    {
        return "camera_link";
    }

    bool finished_;
    std::vector<std::string> joint_names_;

private:

    ros::Publisher marker_array_publisher_;
};


class SkeletonFileStreamAbstract : public SkeletonStream
{
public:

    /**
     *  Get skeleton from input file.
     *  Can be real-time or frame-by-frame.
     */
    SkeletonFileStreamAbstract();
    SkeletonFileStreamAbstract(const std::vector<std::string>& joint_names);

    virtual bool getSkeleton(Eigen::Matrix3Xd& skeleton);

private:

    bool is_realtime_;
};


class SkeletonFileStream : public SkeletonFileStreamAbstract
{
public:

    /**
     *  Get skeleton from input file.
     *
     *  File format
     *   num_joints
     *   joint_names[0] ... joint_names[num_joints-1]
     *   joint_position[0][0] ... joint_position[0][num_joints-1] (where position is 'x y z')
     *   ...
     *   joint_position[?][0] ... joint_position[?][num_joints-1] (until EOF)
     */
    SkeletonFileStream();
    SkeletonFileStream(const std::vector<std::string>& joint_names);
    ~SkeletonFileStream();
    
    inline void setFilename(const std::string& filename)
    {
        filename_ = filename;
    }

    virtual bool getSkeleton(Eigen::Matrix3Xd& skeleton);

    void startReadFrames();

private:

    std::string filename_;

    // file is opened when start requested
    FILE* fp_;
    int file_num_joints_;
    std::vector<std::string> file_joint_names_;
    std::vector<int> file_joint_index_map_;     // joint index in file -> joint index in requested joint names
};


class SkeletonCAD120Stream : public SkeletonFileStreamAbstract
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
        return "camera_depth_frame";
    }

    int getUserId();

    const ros::NodeHandle& nh_;
    tf::TransformListener listener_;
};

}


#endif // PCML_SKELETON_STREAM_H
