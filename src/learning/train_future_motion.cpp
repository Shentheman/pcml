#include <pcml/learning/train_future_motion.h>

#include <pcml/data/skeleton_stream.h>

#include <cassert>


namespace pcml
{

TrainFutureMotion::TrainFutureMotion()
{
    setJointNames(SkeletonStream::jointNamesWholeBody());
    setNumActionTypes(1);
    setT(15);
    setD(15);
}

void TrainFutureMotion::addMotion(const Eigen::MatrixXd& motion, const Eigen::VectorXi action_label)
{
    motions_.push_back(motion);
    action_labels_.push_back(action_label);
}

void TrainFutureMotion::train()
{
    // TODO
}

void TrainFutureMotion::predict(const Eigen::MatrixXd& motion)
{
    predicted_action_probabilities_.resize(num_action_types_, D_);
    predicted_motions_.resize(num_action_types_);
    for (int i=0; i<num_action_types_; i++)
        predicted_motions_[i].resize( joint_names_.size() * 3, D_ );

    // TODO
}

Eigen::MatrixXd TrainFutureMotion::predictedActionProbabilities()
{
    return predicted_action_probabilities_;
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

}
