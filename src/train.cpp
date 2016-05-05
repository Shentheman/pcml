#include <pcml/learning/train_future_motion.h>
#include <pcml/data/skeleton_stream.h>

#include <unistd.h>
#include <stdio.h>

#include <string>

int main(int argc, char** argv)
{
    // parse directory option
    int getopt_res;
    std::string directory;

    getopt_res = getopt(argc, argv, "d:");
    if (getopt_res == -1)
    {
        fprintf(stderr, "Error: cannot find '-d' option.\n");
        fprintf(stderr, "Usage: train -d MODEL_DIRECTORY_PATH\n");
        fflush(stderr);
        return 1;
    }
    directory = optarg;

    pcml::TrainFutureMotion trainer;
    trainer.loadConfig(directory);

    // TODO: add input

    // TODO: train model from input

    trainer.saveTrainedModel("/playpen/jaesungp/indigo_workspace/pcml/trained1");

    return 0;
}

