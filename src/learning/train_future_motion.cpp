#include <pcml/learning/train_future_motion.h>

#include <pcml/data/skeleton_stream.h>

#include <cassert>

#include <algorithm>


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

} // namespace internal


TrainFutureMotion::TrainFutureMotion()
{
    setJointNames(SkeletonStream::jointNamesWholeBody());
    setNumActionTypes(1);
    setT(15);
    setD(15);
    setRBFGamma(1.0);
}

void TrainFutureMotion::addMotion(const Eigen::MatrixXd& motion, const Eigen::VectorXi action_label)
{
    assert(motion.cols() == action_label.rows());

    motions_.push_back(motion);
    action_labels_.push_back(action_label);
}

void TrainFutureMotion::train()
{
    using internal::TrainingInputId;
    using internal::newSvmInput;
    using internal::deleteSvmProblem;

    std::vector<TrainingInputId> training_input_ids;

    // get training input from motions
    for (int i=0; i<motions_.size(); i++)
    {
        // past: (t-T+1)~t, current: t, future: t~(t+D-1) (inclusive)
        // Ex. past: 0~14, current: 14, future: 14~28 (inclusive)
        for (int t = T_ - 1; t + D_ < motions_[i].cols(); t++)
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

    // svm parameters
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.gamma = rbf_gamma_;
    param.cache_size = 100;
    param.eps = 0.001;
    param.C = 1;
    param.nr_weight = 0;
    param.weight_label = 0;
    param.weight = 0;
    param.shrinking = 1;
    param.probability = 1;

    // train current action classifier
    current_action_classifier_ = svm_train( current_action_prob, &param );

    // TODO: train future action classifier
    // TODO: train future motion regressor

    deleteSvmProblem(current_action_prob);
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
    svm_predict_probability( current_action_classifier_, x, const_cast<double*>(predicted_current_action_probabilities_.data()) );

    // TODO: predict future action label
    // TODO: predict future motion

    deleteSvmInput(x);
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
