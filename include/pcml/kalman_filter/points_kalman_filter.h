/*
 * implementation of WAFR 2016 paper
 * prediction based on kalman filter
 */

#ifndef PCML_POINTS_KALMAN_FILTER_H
#define PCML_POINTS_KALMAN_FILTER_H


#include <Eigen/Dense>


namespace pcml
{

class PointsKalmanFilterPredictor
{
public:

    PointsKalmanFilterPredictor();

    void setDeltaT(double delta_t);

    void iterate(const std::vector<Eigen::Vector3d>& z, const std::vector<Eigen::Vector3d> &u);
    void predict(double time, std::vector<Eigen::Matrix<double, 6, 1> >& mu, std::vector<Eigen::Matrix<double, 6, 6> >& sigma);

private:

    void updateControlAndMeasurements(const std::vector<Eigen::Vector3d> &z, const std::vector<Eigen::Vector3d> &u);

    Eigen::Matrix<double, 6, 6> A_;
    Eigen::Matrix<double, 6, 3> B_;
    Eigen::Matrix<double, 3, 6> C_;

    // noises
    Eigen::Matrix<double, 6, 6> R_;
    Eigen::Matrix3d Q_;

    double delta_t_;

    bool first_;

    std::vector<Eigen::Matrix<double, 6, 1> > mus_;
    std::vector<Eigen::Matrix<double, 6, 6> > sigmas_;
};

}


#endif // PCML_POINTS_KALMAN_FILTER_H
