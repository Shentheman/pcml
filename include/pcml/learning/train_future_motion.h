#ifndef PCML_TRAIN_FUTURE_MOTION_H
#define PCML_TRAIN_FUTURE_MOTION_H


#include <vector>
#include <string>
#include <algorithm>

// libsvm
#include <libsvm/svm.h>

#include <eigen3/Eigen/Dense>


namespace pcml
{

class TrainFutureMotion
{
private:

    static const int fps_ = 15;

public:

    TrainFutureMotion();

    inline void setJointNames(const std::vector<std::string>& joint_names, const std::string& root_joint_name = "torso")
    {
        joint_names_ = joint_names;

        root_joint_index_ = std::find(joint_names.begin(), joint_names.end(), root_joint_name) - joint_names.begin();
        if (root_joint_index_ == joint_names.size())
            root_joint_index_ = -1;
    }

    inline void setNumActionTypes(int num_action_types)
    {
        num_action_types_ = num_action_types;
    }

    inline void setT(int T, int T_stride = 1)
    {
        T_ = T;
        T_stride_ = T_stride;
    }

    inline void setD(int D, int D_stride = 1)
    {
        D_ = D;
        D_stride_ = D_stride;
    }

    inline void setRBFGamma(double gamma)
    {
        rbf_gamma_ = gamma;
    }

    inline void setNumSPGPPseudoInputs(int num_spgp_pseudo_inputs)
    {
        num_spgp_pseudo_inputs_ = num_spgp_pseudo_inputs;
    }

    const std::vector<std::string>& jointNames()
    {
        return joint_names_;
    }

    inline int getT()
    {
        return T_;
    }

    /**
     * @brief addMotion Add a training motion.
     * @param motion Motion data of 15 fps as (joint motions as (num_joints * 3) X (num_frames) matrix
     * @param action_label Action labels per frame.
     */
    void addMotion(const Eigen::MatrixXd& motion, const Eigen::VectorXi action_label);

    /**
     * 3 kinds of training
     *
     * @brief train Train using all added training motions and store the trained parameters (for Gaussian Process and SVM classifiers)
     */
    void train();

    /**
     * GP time complexity (estimated)
     *  featuredimperframe * futureframes * actiontype * (examples * featuredimperframe * pastframes + examples * examples)
     *  45 * 5 * 10 * (20 * 45 * 5 + 20 * 20) = 11025000
     *
     * @brief predict Predict and store the future motion distribution (Gaussian distribution) and the future action distrubution (discrete).
     * @param motion The current motion matrix. Only the last T_ frames will be used as input.
     */
    void predict(const Eigen::MatrixXd& motion);

    /**
     * @brief predictedCurrentAction Returns the most likely action label of current time.
     *                               Must be called after predict() function.
     * @return Action label which is the most likely at the current time.
     */
    int predictedCurrentAction();

    /**
     * @brief predictedCurrentActionProbabilities Returns the (softmax) probability distribution of current action labels.
     *                                            result(i) = (probability of action label i)
     *                                            Must be called after predict() function.
     * @return Action label probability distributions at current time.
     */
    Eigen::VectorXd predictedCurrentActionProbabilities();

    /**
     * @brief predictedFutureActionProbabilities Returns the (softmax) probability distribution of future action labels, given current action label guess,
     *                                           result(i,j) = (probability of action label i after (j+1) frames, given current action label guess)
     *                                           Must be called after predict() function.
     * @return Action label probability distributions over future D frames.
     */
    Eigen::MatrixXd predictedFutureActionProbabilities(int action_label);

    /**
     * @brief predictedMotion Returns predicted motion of D frames given the motion and the requested action label.
     *                        m(:,j) = (motion vector after (j+1) frames)
     *                        Must be called after predict() function.
     * @param action_label The action label the predicted motion of which will be returned.
     * @return Predicted motion of requested action label.
     */
    Eigen::MatrixXd predictedMotion(int action_label);

    // load/save
    void loadConfig(std::string directory);
    void saveConfig(std::string directory);
    void loadTrainedModel(std::string directory);
    void saveTrainedModel(std::string directory);

private:

    /// extract feature from input motion.
    /// Input before striding.
    Eigen::VectorXd extractFeature(const Eigen::MatrixXd& motion);

    /// convert future motion to output for learning.
    /// Input before striding.
    Eigen::MatrixXd convertFutureMotion(const Eigen::VectorXd& current_motion, const Eigen::MatrixXd &future_motion);

    // training parameters
    std::vector<std::string> joint_names_;
    int root_joint_index_; // the index of root joint (which is 'torso' in OpenNI tracker)
    int num_action_types_;
    int T_; // # of current frames to be given as input
    int T_stride_;
    int D_; // # of future frames to be predicted
    int D_stride_;
    double rbf_gamma_; // RBF(u, v) = exp(-gamma * |u-v|^2)
    int num_spgp_pseudo_inputs_; // # of pseudo inputs for Sparse Pseudo-input Gaussian Processes (SPGP)

    // training input
    std::vector<Eigen::MatrixXd> motions_;
    std::vector<Eigen::VectorXi> action_labels_;

    // training model
    svm_model* current_action_classifier_;

    // prediction result
    int predicted_current_action_label_;
    Eigen::VectorXd predicted_current_action_probabilities_;
    std::vector<Eigen::MatrixXd> predicted_future_action_probabilities_;
    std::vector<Eigen::MatrixXd> predicted_motions_;
};

}

#endif // PCML_TRAIN_FUTURE_MOTION_H
