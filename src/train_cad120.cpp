#include <pcml/learning/train_future_motion.h>
#include <pcml/data/cad120_reader.h>

#include <unistd.h>
#include <stdio.h>

#include <string>

int main(int argc, char** argv)
{
    // parse directory option
    int getopt_res;
    std::string model_directory = "";
    std::string cad120_directory = "";
    bool cross_validation = false;
    bool grid_search = false;
    bool dfound = false;
    bool cfound = false;

    while ((getopt_res = getopt(argc, argv, "d:c:vg")) != -1)
    {
        switch (getopt_res)
        {
        case 'd':
            dfound = true;
            model_directory = optarg;
            break;

        case 'c':
            cfound = true;
            cad120_directory = optarg;
            break;

        case 'v':
            cross_validation = true;
            break;

        case 'g':
            grid_search = true;
            break;

        default:
            fprintf(stderr, "Usage: train_cad120 -c CAD120_DIRECTORY_PATH -d MODEL_DIRECTORY_PATH [-v] [-g]\n");
            fflush(stderr);
            return 1;
        }
    }

    if (!dfound || !cfound)
    {
        fprintf(stderr, "Usage: train_cad120 -c CAD120_DIRECTORY_PATH -d MODEL_DIRECTORY_PATH [-v] [-g]\n");
        fflush(stderr);
        return 1;
    }

    pcml::TrainFutureMotion trainer(model_directory);
    trainer.loadConfig();

    const std::vector<std::string>& joint_names = trainer.jointNames();

    // add input motions to trainer
    pcml::CAD120Reader reader(cad120_directory);

    for (int subject=0; subject < reader.numSubjects(); subject++)
    {
        for (int action=0; action < reader.numActions(subject); action++)
        {
            for (int video=0; video < reader.numVideos(subject, action); video++)
            {
                // print subject/action/video
                printf("Parsing [subject %d, action %d, video %d]\n", subject, action, video);
                fflush(stdout);

                std::vector<Eigen::VectorXd> motion_list;
                std::vector<int> action_labels_list;

                // reading a demonstration
                reader.startReadFrames(subject, action, video);

                int frame_idx = 0;
                while (reader.readNextFrame())
                {
                    // retrieve motion
                    Eigen::VectorXd current_motion( joint_names.size() * 3 );
                    for (int i=0; i<joint_names.size(); i++)
                    {
                        Eigen::Vector3d position;
                        reader.getJointPosition(joint_names[i], position);

                        current_motion.block( i*3, 0, 3, 1 ) = position;
                    }
                    motion_list.push_back(current_motion);

                    // retrieve action label
                    action_labels_list.push_back( reader.getSubActivityIndex() );

                    /*// print sub-activity per frame
                    printf("sub-activity at %04d: %s\n", frame_idx, reader.getSubActivity().c_str());
                    fflush(stdout);
                    */

                    frame_idx++;
                }

                // conversion from stl vector to eigen matrix/vector
                Eigen::MatrixXd motion( motion_list[0].rows(), motion_list.size() );
                Eigen::VectorXi action_labels( action_labels_list.size() );

                for (int i=0; i<motion_list.size(); i++)
                {
                    motion.col(i) = motion_list[i];
                    action_labels(i) = action_labels_list[i];
                }

                trainer.addMotion(motion, action_labels);
            }
        }
    }

    if (cross_validation)
    {
        printf("Doing cross validation takes a while...\n"); fflush(stdout);
        trainer.crossValidationSVMs();
        printf("Cross validation complete\n"); fflush(stdout);
    }

    if (grid_search)
    {
        printf("Doing grid search takes a while...\n"); fflush(stdout);
        trainer.gridSearchSVMHyperparameters();
        printf("Grid search complete\n"); fflush(stdout);
    }

    if (!cross_validation && !grid_search)
    {
        // training
        printf("Training takes a while...\n"); fflush(stdout);
        trainer.train();
        printf("Training complete\n"); fflush(stdout);

        // saving the trained model
        printf("Saving the trained model\n"); fflush(stdout);
        trainer.saveTrainedModel();
    }

    return 0;
}

