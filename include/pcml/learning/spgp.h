#ifndef PCML_SPGP_H
#define PCML_SPGP_H


#include <eigen3/Eigen/Dense>

// optimization
#include <dlib/optimization.h>


namespace pcml
{

class SparsePseudoinputGaussianProcess
{
private:

    typedef dlib::matrix<double,0,1> column_vector;

    static double var(const Eigen::VectorXd& y);

public:

    SparsePseudoinputGaussianProcess();

    inline void setJitter(const double del)
    {
        del_ = del;
    }

    inline void setNumPseudoinputs(int n)
    {
        num_pseudoinputs_ = n;
    }

    void addInput(const Eigen::VectorXd& x, double y);

    /**
     * Optimizes pseudo-inputs and hyperparameters, and compute intermediate matrices for prediction
     * Hyperparameters are: b(vector), c(scalar), sig(scalar)
     * where K(x, x') = c * exp(-0.5 * sum_d b_d * (x_d - x'_d)^2 + sig * delta(x,x'), delta is Kronecker delta
     */
    void train();

private:

    double kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const Eigen::VectorXd& b, double c, double sig);
    double likelihood(const column_vector& x);
    const column_vector likelihoodDerivative(const column_vector& x);

    void optimizePseudoinputsAndHyperparameters();

    int num_pseudoinputs_;

    double del_; // jitter
    Eigen::MatrixXd X_; // each 'row' corresponds to one input point
    Eigen::VectorXd y_;
    double y_mean_;

    int dim_; // dimension of input point
};

}


#endif // PCML_SPGP_H
