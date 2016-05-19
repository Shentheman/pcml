#include <pcml/learning/train_future_motion.h>
#include <pcml/data/cad120_reader.h>

#include <unistd.h>
#include <stdio.h>

#include <string>

static double standard_error(const Eigen::VectorXd& x)
{
    const int n = x.rows();
    const double mean = x.mean();

    double s = 0.;
    for (int i=0; i<n; i++)
        s += (x(i) - mean) * (x(i) - mean);

    return std::sqrt(s / (n * (n-1)));
}

int main(int argc, char** argv)
{
    // parse directory option
    int getopt_res;
    std::string model_directory = "";
    std::string cad120_directory = "";
    bool dfound = false;
    bool cfound = false;

    while ((getopt_res = getopt(argc, argv, "d:c:")) != -1)
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

        default:
            fprintf(stderr, "Usage: infer_cad120 -c CAD120_DIRECTORY_PATH -d MODEL_DIRECTORY_PATH\n");
            fflush(stderr);
            return 1;
        }
    }

    if (!dfound || !cfound)
    {
        fprintf(stderr, "Usage: infer_cad120 -c CAD120_DIRECTORY_PATH -d MODEL_DIRECTORY_PATH\n");
        fflush(stderr);
        return 1;
    }

    // load trained model
    pcml::TrainFutureMotion trainer(model_directory);
    trainer.loadConfig();
    trainer.loadTrainedModel();

    const int num_action_types = trainer.numActionTypes();
    Eigen::MatrixXd current_action_prediction_count(num_action_types, num_action_types); // [ground_truth][predicted_label] = (count)

    const std::vector<std::string>& joint_names = trainer.jointNames();

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

                Eigen::MatrixXd motion( joint_names.size() * 3, 0 );

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

                    motion.conservativeResize(Eigen::NoChange, motion.cols() + 1);
                    motion.col(motion.cols() - 1) = current_motion;

                    if (frame_idx > trainer.getT())
                    {
                        // retrieve action labels
                        const int ground_truth_sub_activity_label = reader.getSubActivityIndex();

                        trainer.predict(motion);
                        const int predicted_sub_activity_label = trainer.predictedCurrentAction();

                        current_action_prediction_count(ground_truth_sub_activity_label, predicted_sub_activity_label)++;

                        // print sub-activity per frame
                        printf("sub-activity at %04d: %s (truth: %d, prediction: %d)",
                               frame_idx,
                               ground_truth_sub_activity_label == predicted_sub_activity_label ? "True " : "False",
                               ground_truth_sub_activity_label,
                               predicted_sub_activity_label);

                        // print sub-activity probabilities
                        const Eigen::VectorXd probs = trainer.predictedCurrentActionProbabilities();
                        for (int i=0; i<probs.rows(); i++)
                            printf(" %.2lf", probs(i));
                        printf("\n");
                        fflush(stdout);
                    }

                    frame_idx++;
                }
            }
        }
    }

    // performance measurements for current action classification
    Eigen::VectorXd precision = current_action_prediction_count.diagonal().cwiseQuotient( current_action_prediction_count.rowwise().sum() );
    Eigen::VectorXd recall = current_action_prediction_count.diagonal().cwiseQuotient( current_action_prediction_count.colwise().sum().transpose() );

    const double macro_precision = precision.mean();
    const double macro_precision_stderr = standard_error(precision);
    const double macro_recall = recall.mean();
    const double macro_recall_stderr = standard_error(recall);
    const double micro_precision_recall = current_action_prediction_count.trace() / current_action_prediction_count.sum();

    printf("Current action classification confusion matrix:\n");
    for (int i=0; i<num_action_types; i++)
        printf("%s ", reader.getSubActivityFromIndex(i).c_str());
    printf("\n");
    for (int i=0; i<num_action_types; i++)
    {
        const double rowsum = current_action_prediction_count.row(i).sum();
        for (int j=0; j<num_action_types; j++)
            printf("%.2lf ", current_action_prediction_count(i,j) / rowsum);
        printf("\n");
    }
    printf("\n");

    printf("precision: ");
    for (int i=0; i<num_action_types; i++)
        printf("%.2lf ", precision(i));
    printf("\n");

    printf("recall   : ");
    for (int i=0; i<num_action_types; i++)
        printf("%.2lf ", recall(i));
    printf("\n");

    printf("macro-precision: %.2lf +- %.2lf\n", macro_precision, macro_precision_stderr);
    printf("macro-recall   : %.2lf +- %.2lf\n", macro_recall, macro_recall_stderr);
    printf("micro-P/R      : %.2lf\n", micro_precision_recall);

    return 0;
}

