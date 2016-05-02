#include <pcml/learning/train_future_motion.h>

#include <stdio.h>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: train [data path]");
        fflush(stderr);
        return 1;
    }

    pcml::TrainFutureMotion trainer;

    return 0;
}

