#include <pcml/learning/train_future_motion.h>
#include <pcml/data/skeleton_stream.h>

#include <cassert>
#include <algorithm>
#include <fstream>

// yaml-cpp
#include <yaml-cpp/yaml.h>


namespace pcml
{

namespace internal
{

// training input identification
struct TrainingInputId
{
    int motion_index;
    int frame;

    TrainingInputId(int motion_index, int frame)
        : motion_index(motion_index)
        , frame(frame)
    {}
};

svm_node* newSvmInput(const Eigen::VectorXd& f)
{
    int num_nonzero = 0;
    for (int i=0; i<f.rows(); i++)
    {
        if (f(i) != 0.0)
            num_nonzero++;
    }
    svm_node* x = new svm_node[ num_nonzero + 1 ];
    int idx = 0;
    for (int i=0; i<f.rows(); i++)
    {
        if (f(i) != 0.0)
        {
            x[idx].index = i;
            x[idx].value = f(i);
            idx++;
        }
    }
    x[idx].index = -1;

    return x;
}

void deleteSvmInput(svm_node* x)
{
    delete x;
}

void deleteSvmProblem(svm_problem* prob)
{
    for (int i=0; i<prob->l; i++)
        delete prob->x[i];
    delete prob->x;
    delete prob->y;
    delete prob;
}

void removeTrailingSlash(std::string& str)
{
    // remove trailing '/'
    if (*str.rbegin() == '/')
        str = str.substr(0, str.size() - 1);
}

void exportSvmInputToMatlabFormat(const std::string& directory)
{
}

} // namespace internal


TrainFutureMotion::TrainFutureMotion(const std::string& directory)
    : directory_(directory)
{
    setJointNames(SkeletonStream::jointNamesWholeBody());
    setNumActionTypes(1);
    setT(15);
    setD(15);
    setRBFGamma(1.0);
    setSVM_C(1.0);
}

void TrainFutureMotion::addMotion(const Eigen::MatrixXd& motion, const Eigen::VectorXi action_label)
{
    assert(motion.cols() == action_label.rows());

    motions_.push_back(motion);
    action_labels_.push_back(action_label);
}

void TrainFutureMotion::train()
{
    svm_problem* current_action_prob = createSVMProblem();
    svm_parameter param = createSVMParameter();

    // train current action classifier
    const char* error_msg = svm_check_parameter( current_action_prob, &param );
    if (error_msg)
    {
        fprintf(stderr, "SVM parameter Error: %s\n", error_msg);
        fflush(stderr);
        assert(error_msg == 0);
        return;
    }

    current_action_classifier_ = svm_train( current_action_prob, &param );

    // TODO: train future action classifier
    // TODO: train future motion regressor

    // SVM problem should not be freed?
    //deleteSvmProblem(current_action_prob);
}

void TrainFutureMotion::crossValidationSVMs()
{
    svm_problem* current_action_prob = createSVMProblem();
    svm_parameter param = createSVMParameter();

    // shuffle the SVM input (Not sure it affects to cross-validation results)
    std::random_shuffle(&current_action_prob->x[0], &current_action_prob->x[0] + current_action_prob->l);

    // train current action classifier
    const char* error_msg = svm_check_parameter( current_action_prob, &param );
    if (error_msg)
    {
        fprintf(stderr, "SVM parameter Error: %s\n", error_msg);
        fflush(stderr);
        assert(error_msg == 0);
        return;
    }

    double* target = new double[ current_action_prob->l ];
    svm_cross_validation( current_action_prob, &param, 5, target );

    int total_correct = 0;
    for (int i=0; i<current_action_prob->l; i++)
    {
        if (current_action_prob->y[i] == target[i])
            total_correct++;
    }

    printf("Cross validation report:\n");
    printf(" Accuracy: %lf\%\n", (double)total_correct / current_action_prob->l * 100.);
    fflush(stdout);

    delete target;

    // TODO: cross-validate future action classifier

    // SVM problem should not be freed?
    //deleteSvmProblem(current_action_prob);
}

void TrainFutureMotion::gridSearchSVMHyperparameters()
{
    svm_problem* current_action_prob = createSVMProblem();
    svm_parameter param = createSVMParameter();

    // shuffle the SVM input (Not sure it affects to cross-validation results)
    std::random_shuffle(&current_action_prob->x[0], &current_action_prob->x[0] + current_action_prob->l);

    /*// libsvm defalt
    const int c_start = -5;
    const int c_end = 15;
    const int c_step = 2;

    const int gamma_start = -15;
    const int gamma_end = 3;
    const int gamma_step = 2;
    */
    const int c_start = -5;
    const int c_end = 5;
    const int c_step = 2;

    const int gamma_start = -5;
    const int gamma_end = 3;
    const int gamma_step = 2;

    std::vector<double> c_list;
    for (int c = c_start; c <= c_end; c += c_step)
        c_list.push_back((c < 0) ? 1. / (1 << (-c)) : (1 << c));

    std::vector<double> gamma_list;
    for (int g = gamma_start; g <= gamma_end; g += gamma_step)
        gamma_list.push_back((g < 0) ? 1. / (1 << (-g)) : (1 << g));

    std::vector<std::vector<double> > accuracy(c_list.size(), std::vector<double>(gamma_list.size()));

    double* target = new double[ current_action_prob->l ];
    for (int i=0; i<c_list.size(); i++)
    {
        for (int j=0; j<gamma_list.size(); j++)
        {
            const double C = c_list[i];
            const double gamma = gamma_list[j];

            printf("C = %lf, gamma = %lf, ", C, gamma);
            fflush(stdout);

            param.C = C;
            param.gamma = gamma;

            // train current action classifier
            const char* error_msg = svm_check_parameter( current_action_prob, &param );
            if (error_msg)
            {
                fprintf(stderr, "SVM parameter Error: %s\n", error_msg);
                fflush(stderr);
                assert(error_msg == 0);
                return;
            }

            svm_cross_validation( current_action_prob, &param, 5, target );

            int total_correct = 0;
            for (int i=0; i<current_action_prob->l; i++)
            {
                if (current_action_prob->y[i] == target[i])
                    total_correct++;
            }
            const double acc = (double)total_correct / current_action_prob->l * 100.;

            printf("Accuracy: %lf\%\n", acc);
            fflush(stdout);

            accuracy[i][j] = acc;
        }
    }

    delete target;

    printf("C:");
    for (int c = c_start; c <= c_end; c += c_step)
        printf(" 2^%d", c);
    printf("\n");

    printf("G:");
    for (int g = gamma_start; g <= gamma_end; g += gamma_step)
        printf(" 2^%d", g);
    printf("\n");

    printf("                gamma\n");
    printf("                ");
    for (int i=0; i<accuracy[0].size(); i++)
        printf("%8d", i);
    printf("\n");

    for (int i=0; i<accuracy.size(); i++)
    {
        if (i==0) printf("C       ");
        else printf("        ");

        printf("%8d", i);
        for (int j=0; j<accuracy[i].size(); j++)
            printf("%8.2lf", accuracy[i][j]);
        printf("\n");
    }

    // SVM problem should not be freed?
    //deleteSvmProblem(current_action_prob);
}

svm_problem* TrainFutureMotion::createSVMProblem()
{
    using internal::TrainingInputId;
    using internal::newSvmInput;
    using internal::deleteSvmProblem;

    std::vector<TrainingInputId> training_input_ids;

    // get training input from motions
    for (int i=0; i<motions_.size(); i++)
    {
        // skip very near activities
        // past: (t-T+1)~t, current: t, future: t~(t+D-1) (inclusive)
        // Ex. past: 0~14, current: 14, future: 14~28 (inclusive)
        for (int t = T_ - 1; t + D_ < motions_[i].cols(); t += T_ / 2)
            training_input_ids.push_back( TrainingInputId(i, t) );
    }

    // currently all training candidates are used.
    // to use only small set of candidates, randomly shuffle and look up first few ids
    int num_candidates = training_input_ids.size();

    // svm problem specification for current action type classification
    svm_problem* current_action_prob = new svm_problem;
    current_action_prob->l = num_candidates;
    current_action_prob->x = new svm_node* [num_candidates];
    current_action_prob->y = new double [num_candidates];

    for (int i=0; i<num_candidates; i++)
    {
        const int& motion_index = training_input_ids[i].motion_index;
        const int& frame = training_input_ids[i].frame;

        const Eigen::VectorXd f = extractFeature( motions_[motion_index].block(0, frame - T_ + 1, motions_[motion_index].rows(), T_) );
        const int y = action_labels_[motion_index](frame);

        // svm for current action type classification
        current_action_prob->x[i] = newSvmInput(f);
        current_action_prob->y[i] = y;
    }

    return current_action_prob;
}

svm_parameter TrainFutureMotion::createSVMParameter()
{
    // svm parameters
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 1;
    param.gamma = rbf_gamma_;
    param.coef0 = 0.;
    param.cache_size = 100;
    param.eps = 0.001;
    param.C = svm_c_;
    param.nr_weight = 0;
    param.weight_label = 0;
    param.weight = 0;
    param.nu = 0.5;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 1;

    return param;
}

void TrainFutureMotion::predict(const Eigen::MatrixXd& motion)
{
    using internal::newSvmInput;
    using internal::deleteSvmInput;

    // prediction result size initialization
    predicted_current_action_probabilities_.resize(num_action_types_);
    predicted_future_action_probabilities_.resize(num_action_types_);
    predicted_motions_.resize(num_action_types_);
    for (int i=0; i<num_action_types_; i++)
    {
        predicted_future_action_probabilities_[i].resize( num_action_types_, D_ );
        predicted_motions_[i].resize( joint_names_.size() * 3, D_ );
    }

    // input
    Eigen::VectorXd f = extractFeature(motion);
    svm_node* x = newSvmInput(f);

    // predict current action label
    predicted_current_action_label_ = svm_predict_probability( current_action_classifier_, x, const_cast<double*>(predicted_current_action_probabilities_.data()) );

    // TODO: predict future action label
    // TODO: predict future motion

    deleteSvmInput(x);
}

int TrainFutureMotion::predictedCurrentAction()
{
    Eigen::VectorXd::Index row;
    predicted_current_action_probabilities_.maxCoeff(&row, (Eigen::VectorXd::Index*)0);
    return row;
}

Eigen::VectorXd TrainFutureMotion::predictedCurrentActionProbabilities()
{
    return predicted_current_action_probabilities_;
}

Eigen::MatrixXd TrainFutureMotion::predictedFutureActionProbabilities(int action_label)
{
    return predicted_future_action_probabilities_[action_label];
}

Eigen::MatrixXd TrainFutureMotion::predictedMotion(int action_label)
{
    return predicted_motions_[action_label];
}

void TrainFutureMotion::loadConfig()
{
    internal::removeTrailingSlash(directory_);

    // load config in yaml format
    YAML::Node config = YAML::LoadFile( directory_ + "/config.yaml" );

    joint_names_ = config["joint names"].as<std::vector<std::string> >();
    num_action_types_ = config["num action types"].as<int>();
    T_ = config["T"]["value"].as<int>();
    T_stride_ = config["T"]["stride"].as<int>();
    D_ = config["D"]["value"].as<int>();
    D_stride_ = config["D"]["stride"].as<int>();
    rbf_gamma_ = config["RBF gamma"].as<double>();
    svm_c_ = config["SVM C"].as<double>();
    num_spgp_pseudo_inputs_ = config["num SPGP pseudo inputs"].as<int>();
}

void TrainFutureMotion::saveConfig()
{
    internal::removeTrailingSlash(directory_);

    // save config in yaml format
    YAML::Node config;
    config["joint names"] = joint_names_;
    config["num action types"] = num_action_types_;
    config["T"]["value"] = T_;
    config["T"]["stride"] = T_stride_;
    config["D"]["value"] = D_;
    config["D"]["stride"] = D_stride_;
    config["RBF gamma"] = rbf_gamma_;
    config["SVM C"] = svm_c_;
    config["num SPGP pseudo inputs"] = num_spgp_pseudo_inputs_;

    std::ofstream out(directory_ + "/config.yaml");
    out << config;
}

void TrainFutureMotion::loadTrainedModel()
{
    internal::removeTrailingSlash(directory_);

    // load current action label svm classificatio model
    YAML::Node models = YAML::LoadFile( directory_ + "/models.yaml" );

    current_action_classifier_ = svm_load_model( models["current action model"].as<std::string>().c_str() );
}

void TrainFutureMotion::saveTrainedModel()
{
    internal::removeTrailingSlash(directory_);

    // save current action label svm classificatio model
    std::string current_action_classifier_filename = directory_ + "/current_action_model.svm";
    svm_save_model( current_action_classifier_filename.c_str(), current_action_classifier_ );

    // save file list in yaml format
    YAML::Node models;

    models["current action model"] = current_action_classifier_filename;

    std::ofstream out(directory_ + "/models.yaml");
    out << models;
}

Eigen::VectorXd TrainFutureMotion::extractFeature(const Eigen::MatrixXd &motion)
{
    assert(motion.rows() == joint_names_.size() * 3);
    assert(motion.cols() >= T_);

    Eigen::MatrixXd f( motion.rows(), (T_ + T_stride_ - 1) / T_stride_ );

    for (int i=0; i * T_stride_ < T_; i++)
        f.col( f.cols() - i - 1 ) = motion.col( motion.cols() - i * T_stride_ - 1 );

    // make joint positions relative to root joint
    for (int i=0; i<joint_names_.size(); i++)
    {
        if (i != root_joint_index_)
            f.block(i * 3, 0, 3, f.cols()) -= f.block(root_joint_index_ * 3, 0, 3, f.cols());
    }

    // make root joint position at the first frame as origin
    for (int i = f.cols() - 1; i > 0; i--)
        f.block(root_joint_index_ * 3, i, 3, 1) -= f.block(root_joint_index_ * 3, i-1, 3, 1);
    f.block(root_joint_index_ * 3, 0, 3, 1).setZero();

    // reshape as long column vector
    return Eigen::Map<Eigen::VectorXd>(f.data(), f.rows() * f.cols());
}

Eigen::MatrixXd TrainFutureMotion::convertFutureMotion(const Eigen::VectorXd& current_motion, const Eigen::MatrixXd &future_motion)
{
    assert(current_motion.rows() == joint_names_.size() * 3);
    assert(future_motion.rows() == joint_names_.size() * 3);
    assert(future_motion.cols() >= D_);

    Eigen::MatrixXd r( future_motion.rows(), (D_ + D_stride_ - 1) / D_stride_ );

    for (int i=0; i * D_stride_ < D_; i++)
        r.col( r.cols() - i - 1 ) = future_motion.col( D_ - i * D_stride_ - 1 );

    // make joint positions relative to root joint
    for (int i=0; i<joint_names_.size(); i++)
    {
        if (i != root_joint_index_)
            r.block(i * 3, 0, 3, r.cols()) -= r.block(root_joint_index_ * 3, 0, 3, r.cols());
    }

    // make root joint position at the first frame as origin
    for (int i = r.cols() - 1; i > 0; i--)
        r.block(root_joint_index_ * 3, i, 3, 1) -= r.block(root_joint_index_ * 3, i-1, 3, 1);
    r.block(root_joint_index_ * 3, 0, 3, 1) -= current_motion.block(root_joint_index_ * 3, 0, 3, 1);

    return r;
}

} // namespace pcml
