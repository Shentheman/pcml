#include <pcml/learning/spgp.h>

// cholesky decomposition
#include <eigen3/Eigen/Cholesky>

// bind
#include <functional>

// random_shuffle
#include <algorithm>


namespace pcml
{

double SparsePseudoinputGaussianProcess::mean(const Eigen::VectorXd& y)
{
    const int n = y.rows();
    double s = 0;
    for (int i=0; i<n; i++)
        s += y(i);
    return s / n;
}

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
    precomputePredictiveMatrices();
}

void SparsePseudoinputGaussianProcess::optimizePseudoinputsAndHyperparameters()
{
    using namespace std::placeholders;

    column_vector initial_x( (num_pseudoinputs_ + 1) * dim_ + 2);

    std::vector<int> order;
    for (int i=0; i<X_.rows(); i++)
        order.push_back(i);

    // DEBUG: not shuffle data
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

    y0_ = y_.array() - mean(y_);
    const double v = var(y_);
    for (int i=0; i<dim_; i++)
        initial_x(num_pseudoinputs_ * dim_ + i) = -2 * std::log( (X_.col(i).maxCoeff() - X_.col(i).minCoeff()) / 2. );
    initial_x((num_pseudoinputs_ + 1) * dim_) = std::log(v);
    initial_x((num_pseudoinputs_ + 1) * dim_ + 1) = std::log(v / 4.0);

    dlib::find_min(dlib::bfgs_search_strategy(),
                   dlib::objective_delta_stop_strategy(1e-7),
                   std::bind(&SparsePseudoinputGaussianProcess::likelihood, this, _1),
                   std::bind(&SparsePseudoinputGaussianProcess::likelihoodDerivative, this, _1),
                   initial_x,
                   0);

    // the solution is stored in initial_x
    xb_.resize(num_pseudoinputs_, dim_);
    for (int i=0; i<num_pseudoinputs_; i++)
        for (int j=0; j<dim_; j++)
            xb_(i,j) = initial_x(i + j*num_pseudoinputs_);

    b_.resize(dim_);
    for (int i=0; i<dim_; i++)
        b_(i) = std::exp( initial_x(num_pseudoinputs_ * dim_ + i) );

    c_ = std::exp( initial_x((num_pseudoinputs_ + 1) * dim_) );
    sig_ = std::exp( initial_x((num_pseudoinputs_ + 1) * dim_ + 1) );
}

void SparsePseudoinputGaussianProcess::precomputePredictiveMatrices()
{
    /*
    [N,dim] = size(x); n = size(xb,1); Nt = size(xt,1);
    sig = exp(hyp(end)); % noise variance
    */
    const int N = X_.rows();
    const int dim = dim_;

    /*
    % precomputations
    tic;
    K = kern(xb,xb,hyp) + del*eye(n);
    L = chol(K)';
    K = kern(xb,x,hyp);
    V = L\K;
    ep = 1 + (kdiag(x,hyp)-sum(V.^2,1)')/sig;
    V = V./repmat(sqrt(ep)',n,1); y = y./sqrt(ep);
    Lm = chol(sig*eye(n) + V*V')';
    bet = Lm\(V*y);
    clear V
    t_train = toc;
    */

    const int M = xb_.size();
    Eigen::MatrixXd K(M, M);
}

void SparsePseudoinputGaussianProcess::predict()
{
    // need: L, Lm, bet
    /*
    % test predictions
    tic;
    K = kern(xb,xt,hyp);
    lst = L\K;
    clear K
    lmst = Lm\lst;
    mu = (bet'*lmst)';

    s2 = kdiag(xt,hyp) - sum(lst.^2,1)' + sig*sum(lmst.^2,1)';
    t_test = toc;
    */
}

double SparsePseudoinputGaussianProcess::likelihood(const column_vector &w)
{
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
    std::cout << "w = " << std::endl
              << w << std::endl;
    std::cout << "b = " << b.transpose() << std::endl;
    std::cout << "c = " << c << std::endl;
    std::cout << "sig = " << sig << std::endl;
    */

    /*
    xb = xb.*repmat(sqrt(b)',n,1);
    x = x.*repmat(sqrt(b)',N,1);

    Q = xb*xb';
    Q = repmat(diag(Q),1,n) + repmat(diag(Q)',n,1) - 2*Q;
    Q = c*exp(-0.5*Q) + del*eye(n);

    K = -2*xb*x' + repmat(sum(x.*x,2)',n,1) + repmat(sum(xb.*xb,2),1,N);
    K = c*exp(-0.5*K);
    */
    for (int i=0; i<xb.rows(); i++)
        xb.row(i) = xb.row(i).cwiseProduct(b.transpose().array().sqrt().matrix());

    Eigen::MatrixXd x = X_;
    for (int i=0; i<x.rows(); i++)
        x.row(i) = x.row(i).cwiseProduct(b.transpose().array().sqrt().matrix());

    Eigen::MatrixXd Q(n, n);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
            Q(i,j) = c * std::exp( -0.5 * (xb.row(i).transpose() - xb.row(j).transpose()).squaredNorm() );
        Q(i,i) += del_;
    }

    Eigen::MatrixXd K(n, N);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<N; j++)
            K(i,j) = c * std::exp( -0.5 * (xb.row(i).transpose() - x.row(j).transpose()).squaredNorm() );
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

    Eigen::MatrixXd V = L.triangularView<Eigen::Lower>().solve(K);

    Eigen::VectorXd ep = 1 + (c - V.cwiseProduct(V).colwise().sum().array()) / sig;

    Eigen::VectorXd ep_sqrt = ep.array().sqrt();
    for (int i=0; i<K.rows(); i++)
        K.row(i) = K.row(i).cwiseQuotient(ep_sqrt.transpose());

    for (int i=0; i<V.rows(); i++)
        V.row(i) = V.row(i).cwiseQuotient(ep_sqrt.transpose());

    Eigen::VectorXd y = y0_.cwiseQuotient(ep_sqrt);

    Eigen::MatrixXd VVT_sigI = V * V.transpose();
    for (int i=0; i<VVT_sigI.rows(); i++)
        VVT_sigI(i,i) += sig;
    Eigen::LLT<Eigen::MatrixXd> VVT_sigI_cholesky(VVT_sigI);
    Eigen::MatrixXd Lm = VVT_sigI_cholesky.matrixL();
    Eigen::MatrixXd invLmV = Lm.triangularView<Eigen::Lower>().solve(V);
    Eigen::VectorXd bet = invLmV * y;

    /*
    std::cout << "L = " << std::endl
              << L << std::endl;
    std::cout << "V = " << std::endl
              << V << std::endl;
    std::cout << "ep = " << std::endl
              << ep << std::endl;
    std::cout << "K = " << std::endl
              << K << std::endl;
    std::cout << "y = " << std::endl
              << y << std::endl;
    std::cout << "Lm = " << std::endl
              << Lm << std::endl;
    std::cout << "bet = " << std::endl
              << bet << std::endl;
    */

    /*
    % Likelihood
    fw = sum(log(diag(Lm))) + (N-n)/2*log(sig) + ...
          (y'*y - bet'*bet)/2/sig + sum(log(ep))/2 + 0.5*N*log(2*pi);
    */
    double fw = Lm.diagonal().array().log().sum() + (N-n)/2*std::log(sig) +
                (y.transpose() * y - bet.transpose() * bet)(0,0)/2/sig + ep.array().log().sum()/2 + 0.5*N*std::log(2*M_PI);

    /*
    std::cout << "fw = " << fw << std::endl;
    fflush(stdout);
    */

    return fw;
}

const SparsePseudoinputGaussianProcess::column_vector SparsePseudoinputGaussianProcess::likelihoodDerivative(const column_vector &w)
{
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
    xb = xb.*repmat(sqrt(b)',n,1);
    x = x.*repmat(sqrt(b)',N,1);

    Q = xb*xb';
    Q = repmat(diag(Q),1,n) + repmat(diag(Q)',n,1) - 2*Q;
    Q = c*exp(-0.5*Q) + del*eye(n);

    K = -2*xb*x' + repmat(sum(x.*x,2)',n,1) + repmat(sum(xb.*xb,2),1,N);
    K = c*exp(-0.5*K);
    */
    for (int i=0; i<xb.rows(); i++)
        xb.row(i) = xb.row(i).cwiseProduct(b.transpose().array().sqrt().matrix());

    Eigen::MatrixXd x = X_;
    for (int i=0; i<x.rows(); i++)
        x.row(i) = x.row(i).cwiseProduct(b.transpose().array().sqrt().matrix());

    Eigen::MatrixXd Q(n, n);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
            Q(i,j) = c * std::exp( -0.5 * (xb.row(i).transpose() - xb.row(j).transpose()).squaredNorm() );
        Q(i,i) += del_;
    }

    Eigen::MatrixXd K(n, N);
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<N; j++)
            K(i,j) = c * std::exp( -0.5 * (xb.row(i).transpose() - x.row(j).transpose()).squaredNorm() );
    }

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

    Eigen::MatrixXd V = L.triangularView<Eigen::Lower>().solve(K);

    Eigen::VectorXd ep = 1 + (c - V.cwiseProduct(V).colwise().sum().array()) / sig;

    Eigen::VectorXd ep_sqrt = ep.array().sqrt();
    for (int i=0; i<K.rows(); i++)
        K.row(i) = K.row(i).cwiseQuotient(ep_sqrt.transpose());

    for (int i=0; i<V.rows(); i++)
        V.row(i) = V.row(i).cwiseQuotient(ep_sqrt.transpose());

    Eigen::VectorXd y = y0_.cwiseQuotient(ep_sqrt);

    Eigen::MatrixXd VVT_sigI = V * V.transpose();
    for (int i=0; i<VVT_sigI.rows(); i++)
        VVT_sigI(i,i) += sig;
    Eigen::LLT<Eigen::MatrixXd> VVT_sigI_cholesky(VVT_sigI);
    Eigen::MatrixXd Lm = VVT_sigI_cholesky.matrixL();
    Eigen::MatrixXd invLmV = Lm.triangularView<Eigen::Lower>().solve(V);
    Eigen::VectorXd bet = invLmV * y;

    /*
    % precomputations
    Lt = L*Lm;
    B1 = Lt'\(invLmV);
    b1 = Lt'\bet;
    invLV = L'\V;
    invL = inv(L); invQ = invL'*invL; clear invL
    invLt = inv(Lt); invA = invLt'*invLt; clear invLt
    mu = ((Lm'\bet)'*V)';
    sumVsq = sum(V.^2)'; clear V
    bigsum = y.*(bet'*invLmV)'/sig - sum(invLmV.*invLmV)'/2 - (y.^2+mu.^2)/2/sig ...
             + 0.5;
    TT = invLV*(invLV'.*repmat(bigsum,1,n));
    */
    Eigen::MatrixXd Lt = L * Lm;
    Eigen::MatrixXd B1 = Lt.transpose().triangularView<Eigen::Upper>().solve(invLmV);
    Eigen::MatrixXd b1 = Lt.transpose().triangularView<Eigen::Upper>().solve(bet);
    Eigen::MatrixXd invLV = L.transpose().triangularView<Eigen::Upper>().solve(V);
    Eigen::MatrixXd invL = L.inverse();
    Eigen::MatrixXd invQ = invL.transpose() * invL;
    Eigen::MatrixXd invLt = Lt.inverse();
    Eigen::MatrixXd invA = invLt.transpose() * invLt;
    Eigen::VectorXd mu = V.transpose() * (Lm.transpose().triangularView<Eigen::Upper>().solve(bet));
    Eigen::VectorXd sumVsq = V.cwiseProduct(V).colwise().sum().transpose();
    Eigen::VectorXd bigsum = (y.cwiseProduct( (bet.transpose() * invLmV).transpose() ) / sig - invLmV.cwiseProduct(invLmV).colwise().sum().transpose()/2 - (y.cwiseProduct(y) + mu.cwiseProduct(mu))/2/sig).array() + 0.5;

    Eigen::MatrixXd invLVT_bigsum = invLV.transpose();
    for (int i=0; i<invLVT_bigsum.cols(); i++)
        invLVT_bigsum.col(i) = invLVT_bigsum.col(i).cwiseProduct(bigsum);
    Eigen::MatrixXd TT = invLV * invLVT_bigsum;

    /*
    std::cout << "Lt = " << std::endl
              << Lt << std::endl;
    std::cout << "B1 = " << std::endl
              << B1 << std::endl;
    std::cout << "b1 = " << std::endl
              << b1 << std::endl;
    std::cout << "invLV = " << std::endl
              << invLV << std::endl;
    std::cout << "invQ = " << std::endl
              << invQ << std::endl;
    std::cout << "invA = " << std::endl
              << invA << std::endl;
    std::cout << "mu = " << std::endl
              << mu << std::endl;
    std::cout << "sumVsq = " << std::endl
              << sumVsq << std::endl;
    std::cout << "bigsum = " << std::endl
              << bigsum << std::endl;
    std::cout << "TT = " << std::endl
              << TT << std::endl;
    */

    /*
    % pseudo inputs and lengthscales
    for i = 1:dim
    % dnnQ = (repmat(xb(:,i),1,n)-repmat(xb(:,i)',n,1)).*Q;
    % dNnK = (repmat(x(:,i)',n,1)-repmat(xb(:,i),1,N)).*K;
    dnnQ = dist(xb(:,i),xb(:,i)).*Q;
    dNnK = dist(-xb(:,i),-x(:,i)).*K;

    epdot = -2/sig*dNnK.*invLV; epPmod = -sum(epdot)';

    dfxb(:,i) = - b1.*(dNnK*(y-mu)/sig + dnnQ*b1) ...
        + sum((invQ - invA*sig).*dnnQ,2) ...
        + epdot*bigsum - 2/sig*sum(dnnQ.*TT,2);

    dfb(i,1) = (((y-mu)'.*(b1'*dNnK))/sig ...
               + (epPmod.*bigsum)')*x(:,i);

    dNnK = dNnK.*B1; % overwrite dNnK
    dfxb(:,i) = dfxb(:,i) + sum(dNnK,2);
    dfb(i,1) = dfb(i,1) - sum(dNnK,1)*x(:,i);

    dfxb(:,i) = dfxb(:,i)*sqrt(b(i));

    dfb(i,1) = dfb(i,1)/sqrt(b(i));
    dfb(i,1) = dfb(i,1) + dfxb(:,i)'*xb(:,i)/b(i);
    dfb(i,1) = dfb(i,1)*sqrt(b(i))/2;
    end
    */
    Eigen::MatrixXd dfxb(n, dim_);
    Eigen::VectorXd dfb(dim_);
    for (int i=0; i<dim_; i++)
    {
        Eigen::MatrixXd dnnQ(n, n);
        for (int j=0; j<n; j++) for (int k=0; k<n; k++) dnnQ(j,k) = xb(j,i) - xb(k,i);
        dnnQ = dnnQ.cwiseProduct(Q);

        Eigen::MatrixXd dNnK(n, N);
        for (int j=0; j<n; j++) for (int k=0; k<N; k++) dNnK(j,k) = -xb(j,i) - -x(k,i);
        dNnK = dNnK.cwiseProduct(K);

        Eigen::MatrixXd epdot = -2. / sig * dNnK.cwiseProduct(invLV);
        Eigen::VectorXd epPmod = -epdot.colwise().sum().transpose();

        dfxb.col(i) = - b1.cwiseProduct(dNnK * (y-mu) / sig + dnnQ * b1)
                      + (invQ - invA*sig).cwiseProduct(dnnQ).rowwise().sum()
                      + epdot*bigsum - 2. / sig * (dnnQ.cwiseProduct(TT).rowwise().sum());

        dfb(i) = ((y-mu).transpose().cwiseProduct(b1.transpose()*dNnK)/sig + epPmod.cwiseProduct(bigsum).transpose()) * x.col(i);

        dNnK = dNnK.cwiseProduct(B1); // overwrite dNnK
        dfxb.col(i) += dNnK.rowwise().sum();
        dfb(i) -= dNnK.colwise().sum() * x.col(i);

        dfxb.col(i) *= std::sqrt(b(i));

        dfb(i) /= std::sqrt(b(i));
        dfb(i) += (dfxb.col(i).transpose() * xb.col(i))(0,0) / b(i);
        dfb(i) *= std::sqrt(b(i)) / 2.;

        /*
        std::cout << "dnnQ(i) = " << std::endl
                  << dnnQ << std::endl;
        std::cout << "dNnK(i) = " << std::endl
                  << dNnK << std::endl;
        std::cout << "epdot = " << std::endl
                  << epdot << std::endl;
        std::cout << "epPmod = " << std::endl
                  << epPmod << std::endl;
        */
    }

    /*
    std::cout << "dfxb = " << std::endl
              << dfxb << std::endl;
    std::cout << "dfb = " << std::endl
              << dfb << std::endl;
    */

    /*
    % size
    epc = (c./ep - sumVsq - del*sum((invLV).^2)')/sig;

    dfc = (n + del*trace(invQ-sig*invA) ...
         - sig*sum(sum(invA.*Q')))/2 ...
        - mu'*(y-mu)/sig + b1'*(Q-del*eye(n))*b1/2 ...
          + epc'*bigsum;
    */
    Eigen::VectorXd epc = (c * ep.cwiseInverse() - sumVsq - del_ * invLV.cwiseProduct(invLV).colwise().sum().transpose()) / sig;

    double dfc = (n + del_ * (invQ - sig*invA).trace()
                  - sig * invA.cwiseProduct(Q.transpose()).sum()) / 2.
                 - (mu.transpose() * (y - mu) / sig)(0,0) + (b1.transpose() * (Q - del_ * Eigen::MatrixXd::Identity(n, n)) * b1 / 2.)(0,0)
                 + epc.transpose() * bigsum;

    /*
    std::cout << "epc = " << std::endl
              << epc << std::endl;
    */

    /*
    % noise
    dfsig = sum(bigsum./ep);

    dfw = [reshape(dfxb,n*dim,1);dfb;dfc;dfsig];
    */
    double dfsig = bigsum.cwiseQuotient(ep).sum();

    column_vector dfw( (n+1) * dim_ + 2 );
    for (int i=0; i<n; i++)
        for (int j=0; j<dim_; j++)
            dfw(i + j*n) = dfxb(i,j);
    for (int i=0; i<dim_; i++)
        dfw(n*dim_ + i) = dfb(i);
    dfw((n+1)*dim_) = dfc;
    dfw((n+1)*dim_ + 1) = dfsig;

    /*
    std::cout << "dfw = " << std::endl
              << dfw << std::endl;
    */

    return dfw;
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

}
