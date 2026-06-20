#include <Eigen/SVD>
#include <cmath>
#include "8PointAlgorithm.hpp"

Eigen::Matrix3d EightPointAlgorithm::normalizePoints(
    const std::vector<cv::Point2f>& pts,
    std::vector<cv::Point2f>& ptsNorm)
{
    double cx = 0, cy = 0;

    for (const auto& p : pts) {
        cx += p.x;
        cy += p.y;
    }

    cx /= pts.size();
    cy /= pts.size();

    double scale = 0;
    for (const auto& p : pts) {
        scale += std::sqrt((p.x - cx)*(p.x - cx) +
                           (p.y - cy)*(p.y - cy));
    }

    scale = std::sqrt(2.0) * pts.size() / scale;

    ptsNorm.resize(pts.size());

    for (size_t i = 0; i < pts.size(); ++i)
    {
        ptsNorm[i] = cv::Point2f(
            float((pts[i].x - cx) * scale),
            float((pts[i].y - cy) * scale)
        );
    }

    Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
    T(0,0) = scale;
    T(1,1) = scale;
    T(0,2) = -cx * scale;
    T(1,2) = -cy * scale;
    T(2,2) = 1.0;

    return T;
}

Eigen::Matrix3d EightPointAlgorithm::compute(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR)
{
    std::vector<cv::Point2f> Ln, Rn;

    Eigen::Matrix3d Tl = normalizePoints(ptsL, Ln);
    Eigen::Matrix3d Tr = normalizePoints(ptsR, Rn);

    const int n = (int)Ln.size();
    Eigen::MatrixXd A(n, 9);

    for (int i = 0; i < n; ++i)
    {
        double xl = Ln[i].x, yl = Ln[i].y;
        double xr = Rn[i].x, yr = Rn[i].y;

        A.row(i) << xr*xl, xr*yl, xr,
                    yr*xl, yr*yl, yr,
                    xl, yl, 1.0;
    }

    // Smallest right-singular vector of A → f
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd f = svd.matrixV().col(8);

    Eigen::Matrix3d F;
    F << f(0), f(1), f(2),
         f(3), f(4), f(5),
         f(6), f(7), f(8);

    // Enforce rank-2: zero out smallest singular value
    Eigen::JacobiSVD<Eigen::Matrix3d> svdF(
        F, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Vector3d s = svdF.singularValues();
    s(2) = 0.0;

    F = svdF.matrixU() * s.asDiagonal() * svdF.matrixV().transpose();

    // Denormalize: F = Tr^T * F_norm * Tl
    Eigen::Matrix3d Fdenorm =
        Tr.transpose() * F * Tl;

    if (std::abs(Fdenorm(2,2)) > 1e-10)
        Fdenorm /= Fdenorm(2,2);

    return Fdenorm;
}