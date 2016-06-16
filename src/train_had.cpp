#include <pcml/learning/train_future_motion.h>
#include <pcml/data/skeleton_stream.h>

#include <yaml-cpp/yaml.h>

#include <unistd.h>
#include <stdio.h>

#include <string>

int main(int argc, char** argv)
{
#if 0
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    
    // parse directory option
    int getopt_res;
    std::string model_directory = "";
    bool cross_validation = false;
    bool grid_search = false;
    bool dfound = false;

    while ((getopt_res = getopt(argc, argv, "d:c:vg")) != -1)
    {
        switch (getopt_res)
        {
        case 'd':
            dfound = true;
            model_directory = optarg;
            break;

        case 'v':
            cross_validation = true;
            break;

        case 'g':
            grid_search = true;
            break;

        default:
            fprintf(stderr, "Usage: train_had -d MODEL_DIRECTORY_PATH [-v] [-g]\n");
            return 1;
        }
    }

    if (!dfound)
    {
        fprintf(stderr, "Usage: train_had -d MODEL_DIRECTORY_PATH [-v] [-g]\n");
        return 1;
    }

    pcml::TrainFutureMotion trainer(model_directory);
    trainer.loadConfig();
    
    // add input motions to trainer
    const std::vector<std::string>& joint_names = trainer.jointNames();
    pcml::SkeletonFileStream reader(joint_names);

    // metadata
    std::string metadata_filename = model_directory + "/metadata.yaml";
    YAML::Node metadata = YAML::LoadFile(metadata_filename);
    
    const std::string dataset_directory = metadata["directory"].as<std::string>();
    YAML::Node training = metadata["training"];
    for (int i=0; i<training.size(); i++)
    {
        YAML::Node data = training[i];
        const std::string filename = dataset_directory + "/" + data["file"].as<std::string>();
        const int action_label = data["action_label"].as<int>();
        
        // reading a demonstration
        reader.setFilename(filename);
        reader.startReadFrames();

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
            action_labels_list.push_back( action_label );

            /*// print sub-activity per frame
            printf("sub-activity at %04d: %s\n", frame_idx, reader.getSubActivity().c_str());
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

    if (cross_validation)
    {
        printf("Doing cross validation takes a while...\n");
        trainer.crossValidationSVMs();
        printf("Cross validation complete\n");
    }

    if (grid_search)
    {
        printf("Doing grid search takes a while...\n");
        trainer.gridSearchSVMHyperparameters();
        printf("Grid search complete\n");
    }

    if (!cross_validation && !grid_search)
    {
        // training
        printf("Training takes a while...\n");
        trainer.train();
        printf("Training complete\n");

        // saving the trained model
        printf("Saving the trained model\n");
        trainer.saveTrainedModel();
    }

#endif
    return 0;
}

