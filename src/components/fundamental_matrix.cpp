#include <random>
#include <algorithm>
#include "FundamentalMatrix.hpp"
#include "8PointAlgorithm.hpp"

double FundamentalMatrix::sampsonError(
    const Eigen::Matrix3d& F,
    const cv::Point2f& pl,
    const cv::Point2f& pr)
{
    Eigen::Vector3d l(pl.x, pl.y, 1.0);
    Eigen::Vector3d r(pr.x, pr.y, 1.0);

    Eigen::Vector3d Fl  = F * l;
    Eigen::Vector3d Ftr = F.transpose() * r;

    double num = r.dot(Fl);
    double den =
        Fl(0)*Fl(0) + Fl(1)*Fl(1) +
        Ftr(0)*Ftr(0) + Ftr(1)*Ftr(1);

    return (num * num) / den;
}

Eigen::Matrix3d FundamentalMatrix::ransac(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    std::vector<bool>& inlierMask,
    double threshold,
    int maxIter)
{
    const int N = (int)ptsL.size();

    Eigen::Matrix3d bestF = Eigen::Matrix3d::Identity();
    int bestInliers = 0;

    inlierMask.assign(N, false);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);

    for (int it = 0; it < maxIter; ++it)
    {
        std::vector<int> idx;

        while ((int)idx.size() < 8)
        {
            int r = dist(rng);
            if (std::find(idx.begin(), idx.end(), r) == idx.end())
                idx.push_back(r);
        }

        std::vector<cv::Point2f> sL(8), sR(8);

        for (int i = 0; i < 8; ++i)
        {
            sL[i] = ptsL[idx[i]];
            sR[i] = ptsR[idx[i]];
        }

        Eigen::Matrix3d F =
            EightPointAlgorithm::compute(sL, sR);

        std::vector<bool> mask(N);
        int inliers = 0;

        for (int i = 0; i < N; ++i)
        {
            mask[i] =
                sampsonError(F, ptsL[i], ptsR[i]) < threshold;

            if (mask[i]) ++inliers;
        }

        if (inliers > bestInliers)
        {
            bestInliers = inliers;
            bestF = F;
            inlierMask = mask;
        }
    }

    // Refit
    std::vector<cv::Point2f> inL, inR;

    for (int i = 0; i < N; ++i)
    {
        if (inlierMask[i])
        {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
        }
    }

    if (inL.size() >= 8)
    {
        bestF = EightPointAlgorithm::compute(inL, inR);

        for (int i = 0; i < N; ++i)
        {
            inlierMask[i] =
                sampsonError(bestF, ptsL[i], ptsR[i]) < threshold;
        }
    }

    return bestF;
}