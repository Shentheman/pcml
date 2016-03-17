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

    CAD120Reader();
    CAD120Reader(const std::string& directory);

    // should not contain symbolic links
    void setDirectory(const std::string& directory);

    int numSubjects();
    int numActions(int subject);
    int numVideos(int subject, int action);

    // rgbd frames
    void startReadFrames(int subject, int action, int video);
    bool readNextFrame();
    void finishReadFrames();

    // debug
    void print();

    // visualize
    void setPointCloudTopic(const std::string& topic);
    void renderPointCloud();

private:

    void readActionInfo(Action& action, const std::string& action_directory);

    // ros publishers
    ros::Publisher point_cloud_publisher_;

    // CAD120 dataset root directory
    std::string directory_;

    // annotations
    std::vector<Subject> subjects_;

    // rgbd data
    FILE* rgbd_fp_;
    int rgbd_image_[Y_RES][X_RES][4];
    int subject_;
    int action_;
    int video_;
};

}


#endif // PCML_CAD120_READER_H
