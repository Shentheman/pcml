#include <pcml/learning/train_future_motion.h>
#include <pcml/data/skeleton_stream.h>

#include <ros/ros.h>

int main(int argc, char** argv)
{
    pcml::TrainFutureMotion trainer;
    trainer.setJointNames(pcml::SkeletonStream::jointNamesUpperBody());
    trainer.setNumActionTypes(10);
    trainer.setT(15);
    trainer.setD(15);

    return 0;
}

