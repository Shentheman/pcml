#include <ros/ros.h>

#include <pcml/data/skeleton_stream.h>


/**
 *  roslaunch openni_launch openni.launch
 *  rosrun openni_tracker openni.tracker
 *
 *  Specify path of directory containing CAD120 dataset
 */
int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_skeleton_stream");

    ros::NodeHandle nh;
    ros::Rate rate(30);

    pcml::SkeletonCAD120Stream cad120_stream("/playpen/jaesungp/dataset/CAD120");
    cad120_stream.setVisualizationTopic(nh, "skeleton_cad120");

    pcml::SkeletonRealtimeStream realtime_stream(nh);
    realtime_stream.setVisualizationTopic(nh, "skeleton_realtime");

    const int subject = 0;
    const int action = 0;
    const int video = 0;

    // cad120 stream
    cad120_stream.startReadFrames(subject, action, video);
    while (!cad120_stream.isFinished() && ros::ok())
    {
        cad120_stream.renderSkeleton();

        rate.sleep();
    }

    // realtime stream
    while (ros::ok())
    {
        realtime_stream.renderSkeleton();

        rate.sleep();
    }

    ros::shutdown();
    return 0;
}
