#include <ros/ros.h>

#include <pcml/data/cad120_reader.h>


int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_cad120");

    ros::NodeHandle nh;
    ros::Rate rate(60);

    pcml::CAD120Reader reader("/playpen/jaesungp/dataset/CAD120/");
    reader.print();

    reader.setPointCloudTopic("CAD120");
    reader.setMarkerArrayTopic("CAD120_marker");
    ros::Duration(1.0).sleep();

    for (int subject=0; subject < reader.numSubjects(); subject++)
    {
        for (int action=0; action < reader.numActions(subject); action++)
        {
            for (int video=0; video < reader.numVideos(subject, action); video++)
            {
                printf("subject %d, action %d, video %d\n", subject, action, video);
                fflush(stdout);

                reader.startReadFrames(subject, action, video);

                while (reader.readNextFrame())
                {
                    reader.renderPointCloud();
                    reader.renderSkeleton();
                    reader.renderObjects();
                    rate.sleep();
                }
            }
        }
    }

    return 0;
}
