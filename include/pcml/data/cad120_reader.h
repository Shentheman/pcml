#ifndef PCML_CAD120_READER_H
#define PCML_CAD120_READER_H


#include <string>
#include <vector>

#include <Eigen/Dense>

#include <ros/ros.h>


namespace pcml
{

class CAD120Reader
{
private:

    static const int X_RES = 640;
    static const int Y_RES = 480;

    static const int num_joints_ = 15;
    static const int num_joints_with_ori_ = 11;   // only the first 11 joints have orientations
    static const int num_skeletons_ = 14;
    static const char* joint_names_[num_joints_];
    static const int skeletons_[num_skeletons_][2];

    struct FrameIntervalAnnotation
    {
        int start_frame;
        int end_frame;
        std::string sub_activity_id;
        std::vector<std::string> affordance_ids;
    };

    struct Video
    {
        std::string id;
        std::string activity_id;
        std::string subject_id;
        std::vector<int> object_ids;
        std::vector<std::string> object_types;
        Eigen::Matrix4d global_transform;
        std::vector<FrameIntervalAnnotation> annotations;
    };

    struct Action
    {
        std::string action_id;
        std::vector<Video> videos;
    };

    struct Subject
    {
        int subject_id;
        std::vector<Action> actions;
    };

    static bool existDirectory(const std::string& directory);

public:

    static int numJoints()
    {
        return num_joints_;
    }

public:

    CAD120Reader();
    CAD120Reader(const std::string& directory);

    // should not contain symbolic links
    void setDirectory(const std::string& directory);

    int numSubjects();
    int numActions(int subject);
    int numVideos(int subject, int action);

    // data retrieval
    bool getJointPosition(const std::string& joint_name, Eigen::Vector3d& position);

    // rgbd frames
    void startReadFrames(int subject, int action, int video);
    bool readNextFrame();
    void finishReadFrames();

    // debug
    void print();

    // visualize
    void setPointCloudTopic(const std::string& topic);
    void renderPointCloud();
    void setMarkerArrayTopic(const std::string& topic);
    void renderSkeleton();
    void renderObjects();

private:

    void readActionInfo(Action& action, const std::string& action_directory);

    // find depths of certain percentages inside the bounding box in the depth image
    std::vector<int> findDepths(const Eigen::Vector4d& bounding_box, const std::vector<double>& percentages);

    // ros publishers
    ros::Publisher point_cloud_publisher_;
    ros::Publisher marker_array_publisher_;

    // CAD120 dataset root directory
    std::string directory_;

    // non-per-frame data
    std::vector<Subject> subjects_;

    // per-frame rgbd data
    FILE* rgbd_fp_;
    int rgbd_image_[Y_RES][X_RES][4];
    int subject_;
    int action_;
    int video_;

    // per-frame skeleton data
    // It retains the original data.
    // In current implementation, the transformation is applied only in the visualization function.
    FILE* joint_fp_;
    Eigen::Matrix3d joint_ori_[num_joints_];
    Eigen::Vector3d joint_pos_[num_joints_];
    bool joint_ori_confidence_[num_joints_];
    bool joint_pos_confidence_[num_joints_];

    // per-frame object annotation data
    std::vector<FILE*> object_fps_;
    std::vector<Eigen::Vector4d> object_bounding_boxes_; // (x, y) of upper-left, (x, y) of lower right
    std::vector<Eigen::Matrix<double, 2, 3> > object_transformations_; // SIFT feature transformation from previous frame
};

}


#endif // PCML_CAD120_READER_H
