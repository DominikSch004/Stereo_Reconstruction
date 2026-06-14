#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

// Hartley normalization: zero mean, average distance sqrt(2)
inline Eigen::Matrix3d normalizePoints(
    const std::vector<cv::Point2f>& pts,
    std::vector<cv::Point2f>& ptsNorm)
{
    double cx = 0, cy = 0;
    for (const auto& p : pts) { cx += p.x; cy += p.y; }
    cx /= pts.size(); cy /= pts.size();

    double scale = 0;
    for (const auto& p : pts)
        scale += std::sqrt((p.x - cx)*(p.x - cx) + (p.y - cy)*(p.y - cy));
    scale = std::sqrt(2.0) * pts.size() / scale;

    ptsNorm.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
        ptsNorm[i] = { float((pts[i].x - cx) * scale), float((pts[i].y - cy) * scale) };

    Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
    T(0,0) = scale; T(0,2) = -cx * scale;
    T(1,1) = scale; T(1,2) = -cy * scale;
    T(2,2) = 1.0;
    return T;
}

// Normalized 8-point algorithm → Fundamental matrix
inline Eigen::Matrix3d eightPoint(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR)
{
    std::vector<cv::Point2f> ptsLn, ptsRn;
    Eigen::Matrix3d Tl = normalizePoints(ptsL, ptsLn);
    Eigen::Matrix3d Tr = normalizePoints(ptsR, ptsRn);

    int n = ptsLn.size();
    Eigen::MatrixXd A(n, 9);
    for (int i = 0; i < n; ++i)
    {
        double xl = ptsLn[i].x, yl = ptsLn[i].y;
        double xr = ptsRn[i].x, yr = ptsRn[i].y;
        A.row(i) << xr*xl, xr*yl, xr, yr*xl, yr*yl, yr, xl, yl, 1.0;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeFullV);
    Eigen::VectorXd f = svdA.matrixV().col(8);
    Eigen::Matrix3d F;
    F << f(0), f(1), f(2),
         f(3), f(4), f(5),
         f(6), f(7), f(8);

    // Enforce rank-2
    Eigen::JacobiSVD<Eigen::Matrix3d> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s = svdF.singularValues();
    s(2) = 0.0;
    F = svdF.matrixU() * s.asDiagonal() * svdF.matrixV().transpose();

    F = Tr.transpose() * F * Tl;
    if (std::abs(F(2,2)) > 1e-10) F /= F(2,2);
    return F;
}

// Sampson distance
inline double sampsonError(
    const Eigen::Matrix3d& F,
    const cv::Point2f& pl, const cv::Point2f& pr)
{
    Eigen::Vector3d l(pl.x, pl.y, 1.0), r(pr.x, pr.y, 1.0);
    Eigen::Vector3d Fl  = F * l;
    Eigen::Vector3d Ftr = F.transpose() * r;
    double num = r.dot(Fl);
    double den = Fl(0)*Fl(0) + Fl(1)*Fl(1) + Ftr(0)*Ftr(0) + Ftr(1)*Ftr(1);
    return (num * num) / den;
}

// RANSAC + 8-point → Fundamental matrix
inline Eigen::Matrix3d ransacFundamental(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    std::vector<bool>& inlierMask,
    double threshold = 1.0,
    int maxIter = 1000)
{
    const int N = ptsL.size();
    Eigen::Matrix3d bestF = Eigen::Matrix3d::Identity();
    int bestInliers = 0;
    inlierMask.assign(N, false);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);

    for (int iter = 0; iter < maxIter; ++iter)
    {
        std::vector<int> idx;
        idx.reserve(8);
        while ((int)idx.size() < 8)
        {
            int s = dist(rng);
            if (std::find(idx.begin(), idx.end(), s) == idx.end())
                idx.push_back(s);
        }

        std::vector<cv::Point2f> sL(8), sR(8);
        for (int i = 0; i < 8; ++i) { sL[i] = ptsL[idx[i]]; sR[i] = ptsR[idx[i]]; }

        Eigen::Matrix3d F = eightPoint(sL, sR);

        std::vector<bool> mask(N);
        int inliers = 0;
        for (int i = 0; i < N; ++i)
        {
            mask[i] = sampsonError(F, ptsL[i], ptsR[i]) < threshold;
            if (mask[i]) ++inliers;
        }

        if (inliers > bestInliers)
        {
            bestInliers = inliers;
            bestF = F;
            inlierMask = mask;
        }
    }

    // Refit on all inliers
    std::vector<cv::Point2f> inL, inR;
    for (int i = 0; i < N; ++i)
        if (inlierMask[i]) { inL.push_back(ptsL[i]); inR.push_back(ptsR[i]); }

    if ((int)inL.size() >= 8)
    {
        bestF = eightPoint(inL, inR);
        for (int i = 0; i < N; ++i)
            inlierMask[i] = sampsonError(bestF, ptsL[i], ptsR[i]) < threshold;
    }

    return bestF;
}
