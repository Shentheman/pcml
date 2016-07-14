#include <pcml/kalman_filter/points_kalman_filter.h>


namespace pcml
{

PointsKalmanFilterPredictor::PointsKalmanFilterPredictor()
    : first_(true)
{
    A_.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
    A_.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity();

    B_.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();

    C_.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity();
    C_.block(0, 3, 3, 3) = Eigen::Matrix3d::Zero();

    setDeltaT(1. / 15);
}

void PointsKalmanFilterPredictor::setDeltaT(double delta_t)
{
    delta_t_ = delta_t;

    A_.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity() * delta_t;

    B_.block(3, 0, 3, 3) = Eigen::Matrix3d::Identity() * delta_t;
}

void PointsKalmanFilterPredictor::iterate(const std::vector<Eigen::Vector3d> &z, const std::vector<Eigen::Vector3d> &u)
{
    if (first_)
    {
        first_ = false;

        mus_.resize(z.size());
        sigmas_.resize(z.size());

        for (int i=0; i<z.size(); i++)
        {
            mus_[i].block(0, 0, 3, 1) = z[i];
            mus_[i].block(3, 0, 3, 1).setZero();

            sigmas_[i].block(0, 0, 3, 3) = Q_;
            sigmas_[i].block(0, 0, 3, 3) = 2. * Q_ / (delta_t_ * delta_t_);
        }
    }
    else
    {
        updateControlAndMeasurements(z, u);
    }
}

void PointsKalmanFilterPredictor::updateControlAndMeasurements(const std::vector<Eigen::Vector3d> &z, const std::vector<Eigen::Vector3d> &u)
{
    for (int i=0; i<mus_.size(); i++)
    {
        mus_[i] = A_ * mus_[i] + B_ * u[i];
        sigmas_[i] = A_ * sigmas_[i] * A_.transpose() + R_;
        Eigen::MatrixXd K = sigmas_[i] * C_.transpose() * (C_ * sigmas_[i] * C_.transpose() + Q_).inverse();
        mus_[i] = mus_[i] + K * (z[i] - C_ * mus_[i]);
        sigmas_[i] = (Eigen::Matrix<double, 6, 6>::Identity() - K * C_) * sigmas_[i];
    }
}

void PointsKalmanFilterPredictor::predict(double time, std::vector<Eigen::Matrix<double, 6, 1> >& mu, std::vector<Eigen::Matrix<double, 6, 6> >& sigma)
{
    if (time == 0.)
    {
        mu = mus_;
        sigma = sigmas_;
    }
    else
    {
        Eigen::Matrix<double, 6, 6> A;
        Eigen::Matrix<double, 6, 3> B;

        A.setIdentity();
        A.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity() * time;

        B.block(0, 0, 3, 3).setIdentity();
        B.block(3, 0, 3, 3) = Eigen::Matrix3d::Identity() * time;

        mu.resize(mus_.size());
        sigma.resize(sigmas_.size());

        for (int i=0; i<mu.size(); i++)
        {
            mu[i] = A * mus_[i]; // assuming control input u = 0. control input should be dealt with in caller
            sigma[i] = A * sigmas_[i] * A.transpose() + R_;
        }
    }
}

}
