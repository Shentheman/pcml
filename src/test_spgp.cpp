#include <stdio.h>
#include <sstream>
#include <iostream>

#include <stdlib.h>
#include <time.h>

#include <eigen3/Eigen/Dense>

#include <pcml/learning/spgp.h>

int main()
{
    srand(time(0));

    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::MatrixXd Xt;

    static char buffer[1024];

    FILE* fp = fopen("../spgp_data/train_inputs", "r");
    while (true)
    {
        if (fgets(buffer, 1024, fp) == NULL || buffer[0] == '\n' || buffer[0] == '\r')
            break;

        std::istringstream is(buffer);
        Eigen::VectorXd x;
        while (!is.eof())
        {
            double v;
            is >> v;
            if (is.fail()) break;
            x.conservativeResize(x.rows() + 1);
            x(x.rows() - 1) = v;
        }
        X.conservativeResize(X.rows() + 1, x.rows());
        X.row(X.rows() - 1) = x.transpose();
    }
    fclose(fp);

    fp = fopen("../spgp_data/train_outputs", "r");
    while (true)
    {
        if (fgets(buffer, 1024, fp) == NULL || buffer[0] == '\n' || buffer[0] == '\r')
            break;

        y.conservativeResize(y.rows() + 1);
        sscanf(buffer, "%lf", &y(y.rows() - 1));
    }
    fclose(fp);

    fp = fopen("../spgp_data/test_inputs", "r");
    while (true)
    {
        if (fgets(buffer, 1024, fp) == NULL || buffer[0] == '\n' || buffer[0] == '\r')
            break;

        std::istringstream is(buffer);
        Eigen::VectorXd x;
        while (!is.eof())
        {
            double v;
            is >> v;
            if (is.fail()) break;
            x.conservativeResize(x.rows() + 1);
            x(x.rows() - 1) = v;
        }
        Xt.conservativeResize(Xt.rows() + 1, x.rows());
        Xt.row(Xt.rows() - 1) = x.transpose();
    }
    fclose(fp);

    pcml::SparsePseudoinputGaussianProcess spgp;
    spgp.setJitter(1e-6);
    spgp.setNumPseudoinputs(20);

    for (int i=0; i<X.rows(); i++)
        spgp.addInput(X.row(i), y(i));

    spgp.train();

    fflush(stdout);
    return 0;
}
