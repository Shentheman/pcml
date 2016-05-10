#include <pcml/learning/spgp.h>

// cholesky decomposition
#include <eigen3/Eigen/Cholesky>

// bind
#include <functional>

// random_shuffle
#include <algorithm>


namespace pcml
{

double SparsePseudoinputGaussianProcess::var(const Eigen::VectorXd& y)
{
    const int n = y.rows();
    double s = 0, ss=0;
    for (int i=0; i<n; i++)
    {
        s += y(i);
        ss += y(i) * y(i);
    }
    return (ss - s * s / n) / n;
}

SparsePseudoinputGaussianProcess::SparsePseudoinputGaussianProcess()
{
}

void SparsePseudoinputGaussianProcess::addInput(const Eigen::VectorXd &x, double y)
{
    if (X_.rows() == 0)
    {
        dim_ = x.rows();

        X_ = x.transpose();
        y_.resize(1);
        y_(0) = y;
    }
    else
    {
        X_.conservativeResize(X_.rows() + 1, Eigen::NoChange);
        X_.row( X_.rows()-1 ) = x.transpose();
        y_.conservativeResize(y_.rows() + 1);
        y_( y_.rows()-1 ) = y;
    }
}

void SparsePseudoinputGaussianProcess::train()
{
    optimizePseudoinputsAndHyperparameters();
}

void SparsePseudoinputGaussianProcess::optimizePseudoinputsAndHyperparameters()
{
    using namespace std::placeholders;

    column_vector initial_x( (num_pseudoinputs_ + 1) * dim_ + 2);

    std::vector<int> order;
    for (int i=0; i<X_.rows(); i++)
        order.push_back(i);
    std::random_shuffle(order.begin(), order.end());

    for (int j=0; j<dim_; j++)
    {
        for (int i=0; i<num_pseudoinputs_; i++)
            initial_x(i + j*num_pseudoinputs_) = X_(order[i], j);
    }

    /*
    hyp_init(1:dim,1) = -2*log((max(x)-min(x))'/2); % log 1/(lengthscales)^2
    hyp_init(dim+1,1) = log(var(y0,1)); % log size
    hyp_init(dim+2,1) = log(var(y0,1)/4); % log noise
    */
    const double v = var(y_);
    for (int i=0; i<dim_; i++)
        initial_x(num_pseudoinputs_ * dim_ + i) = -2 * std::log( (X_.col(i).maxCoeff() - X_.col(i).minCoeff()) / 2. );
    initial_x((num_pseudoinputs_ + 1) * dim_) = std::log(v);
    initial_x((num_pseudoinputs_ + 1) * dim_ + 1) = std::log(v / 4.0);

    likelihood(initial_x);
    return;

    dlib::find_min(dlib::bfgs_search_strategy(),
                   dlib::objective_delta_stop_strategy(1e-7),
                   std::bind(&SparsePseudoinputGaussianProcess::likelihood, this, _1),
                   std::bind(&SparsePseudoinputGaussianProcess::likelihoodDerivative, this, _1),
                   initial_x,
                   0);
}

double SparsePseudoinputGaussianProcess::likelihood(const column_vector &w)
{
    double f;

    /*
    [N,dim] = size(x); xb = reshape(w(1:end-dim-2),n,dim);
    b = exp(w(end-dim-1:end-2)); c = exp(w(end-1)); sig = exp(w(end));
    */
    const int N = X_.rows();
    const int n = num_pseudoinputs_;

    Eigen::MatrixXd xb(n, dim_);
    for (int j=0; j<dim_; j++)
    {
        for (int i=0; i<n; i++)
            xb(i,j) = w(i + j*n);
    }

    Eigen::VectorXd b(dim_);
    for (int i=0; i<dim_; i++)
        b(i) = std::exp( w(n * dim_ + i) );

    const double c = std::exp( w((n+1) * dim_) );
    const double sig = std::exp( w((n+1) * dim_ + 1) );

    /*
    Q = xb*xb';
    Q = repmat(diag(Q),1,n) + repmat(diag(Q)',n,1) - 2*Q;
    Q = c*exp(-0.5*Q) + del*eye(n);

    K = -2*xb*x' + repmat(sum(x.*x,2)',n,1) + repmat(sum(xb.*xb,2),1,N);
    K = c*exp(-0.5*K);

    % Km = Q;
    % Kmn = K;
    */
    Eigen::MatrixXd Q(n, n);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
            Q(i,j) = kernel(xb.row(i).transpose(), xb.row(j).transpose(), b, c, sig);
    }

    Eigen::MatrixXd K(n, N);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<N; j++)
            K(i,j) = kernel(xb.row(i).transpose(), X_.row(j).transpose(), b, c, sig);
    }

    /*
    std::cout << "Q = " << std::endl
              << Q << std::endl;
    std::cout << "K = " << std::endl
              << K << std::endl;
    */

    /*
    L = chol(Q)'; % MxM, M^3
    V = L\K; % MxN, M^2 N
    ep = 1 + (c-sum(V.^2)')/sig; % N, MN
    K = K./repmat(sqrt(ep)',n,1); % MxN, MN
    V = V./repmat(sqrt(ep)',n,1); y = y./sqrt(ep); % MxN, MN
    Lm = chol(sig*eye(n) + V*V')'; % MxM, M^3
    invLmV = Lm\V; % MxN, M^2 N
    bet = invLmV*y; % M, MN
    */
    Eigen::LLT<Eigen::MatrixXd> Q_cholesky(Q);
    Eigen::MatrixXd L = Q_cholesky.matrixL();

    Eigen::MatrixXd V = Q_cholesky.solve(K);

    Eigen::VectorXd ep = V.colwise().sum();

    for (int i=0; i<K.rows(); i++)
        K.row(i) = K.row(i).cwiseQuotient(ep.transpose().sqrt());

    return f;
}

double SparsePseudoinputGaussianProcess::kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const Eigen::VectorXd& b, double c, double sig)
{
    double s = 0;
    for (int i=0; i<dim_; i++)
    {
        const double d = x1(i) - x2(i);
        s += d*d * b(i);
    }

    return c * std::exp( -0.5 * s );
}

const SparsePseudoinputGaussianProcess::column_vector SparsePseudoinputGaussianProcess::likelihoodDerivative(const column_vector &w)
{
    column_vector d;

    return d;
}

}
