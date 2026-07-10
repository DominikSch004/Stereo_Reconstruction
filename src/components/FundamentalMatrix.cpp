#include "FundamentalMatrix.hpp"
#include <opencv2/calib3d.hpp>
#include <Eigen/SVD>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

double FundamentalMatrix::sampsonError(
    const Eigen::Matrix3d &F,
    const cv::Point2f &pl,
    const cv::Point2f &pr)
{
    // Convert 2D pixel coordinates to homogeneous 3D vectors
    Eigen::Vector3d l(pl.x, pl.y, 1.0);
    Eigen::Vector3d r(pr.x, pr.y, 1.0);

    // Compute epipolar lines
    Eigen::Vector3d Fl = F * l;
    Eigen::Vector3d Ftr = F.transpose() * r;

    // Algebraic error: r^T * F * l
    double num = r.dot(Fl);

    // sum of squared entry gradients for both images (denominator of the Sampson distance)
    double den = Fl(0) * Fl(0) + Fl(1) * Fl(1) +
                 Ftr(0) * Ftr(0) + Ftr(1) * Ftr(1);

    // first-order geometric approximation of the epipolar distance
    return (num * num) / den;
}

Eigen::Matrix3d FundamentalMatrix::normalizePoints(
    const std::vector<cv::Point2f> &pts,
    std::vector<cv::Point2f> &ptsNorm)
{
    // centroid (mean x, y) of the 2D point cloud
    double cx = 0, cy = 0;
    for (const auto &p : pts)
    {
        cx += p.x;
        cy += p.y;
    }
    cx /= pts.size();
    cy /= pts.size();

    // average distance from the shifting center point
    double scale = 0;
    for (const auto &p : pts)
    {
        scale += std::sqrt((p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy));
    }

    // Scale factor sets average distance of transformed points to sqrt(2)
    scale = std::sqrt(2.0) * pts.size() / scale;

    // Shift points to centroid and scale
    ptsNorm.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
    {
        ptsNorm[i] = cv::Point2f(
            float((pts[i].x - cx) * scale),
            float((pts[i].y - cy) * scale));
    }

    // 3x3 transformation matrix T: p_norm = T * p
    Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
    T(0, 0) = scale;
    T(1, 1) = scale;
    T(0, 2) = -cx * scale;
    T(1, 2) = -cy * scale;
    T(2, 2) = 1.0;

    return T;
}

Eigen::Matrix3d FundamentalMatrix::compute8Point(
    const std::vector<cv::Point2f> &ptsL,
    const std::vector<cv::Point2f> &ptsR)
{
    // Hartley normalization to ensure numerical stability
    std::vector<cv::Point2f> Ln, Rn;
    Eigen::Matrix3d Tl = normalizePoints(ptsL, Ln);
    Eigen::Matrix3d Tr = normalizePoints(ptsR, Rn);

    // homogeneous linear system matrix A from epipolar constraint equations
    const int n = (int)Ln.size();
    Eigen::MatrixXd A(n, 9);

    for (int i = 0; i < n; ++i)
    {
        double xl = Ln[i].x, yl = Ln[i].y;
        double xr = Rn[i].x, yr = Rn[i].y;
        A.row(i) << xr * xl, xr * yl, xr, yr * xl, yr * yl, yr, xl, yl, 1.0;
    }

    // solution vector f is the last column of V (smallest singular value)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd f = svd.matrixV().col(8);

    Eigen::Matrix3d F;
    F << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);

    // Enforce the Singularity Constraint (Rank-2 condition, det(F) = 0)
    Eigen::JacobiSVD<Eigen::Matrix3d> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s = svdF.singularValues();
    s(2) = 0.0; // Force smallest singular value to zero to collapse the rank
    F = svdF.matrixU() * s.asDiagonal() * svdF.matrixV().transpose();

    // Denormalize: map F_norm back to pixel space coords using transformation matrices
    Eigen::Matrix3d Fdenorm = Tr.transpose() * F * Tl;

    // Normalize last matrix element to 1 for standard scaling consistency
    if (std::abs(Fdenorm(2, 2)) > 1e-10)
    {
        Fdenorm /= Fdenorm(2, 2);
    }
    return Fdenorm;
}

Eigen::Matrix3d FundamentalMatrix::computeFundamental(
    const std::vector<cv::Point2f> &ptsL,
    const std::vector<cv::Point2f> &ptsR,
    std::vector<bool> &inlierMask,
    FundamentalMethod method,
    double threshold,
    double confidence,
    int maxIter)
{
    switch (method)
    {
    case FundamentalMethod::CustomRANSAC:
        return computeCustomRANSAC(ptsL, ptsR, inlierMask, confidence, threshold, maxIter);
    case FundamentalMethod::OpenCVRANSAC:
        return computeOpenCVRANSAC(ptsL, ptsR, inlierMask, threshold, confidence);
    case FundamentalMethod::CustomMAGSAC:
        return computeCustomMAGSAC(ptsL, ptsR, inlierMask, threshold, confidence, maxIter);
    case FundamentalMethod::CustomPROSAC:
        return computeCustomPROSAC(ptsL, ptsR, inlierMask, threshold, confidence, maxIter);
    }

    inlierMask.assign(ptsL.size(), false);
    return Eigen::Matrix3d::Identity();
}

Eigen::Matrix3d FundamentalMatrix::computeCustomRANSAC(
    const std::vector<cv::Point2f> &ptsL,
    const std::vector<cv::Point2f> &ptsR,
    std::vector<bool> &inlierMask,
    double confidence,
    double threshold,
    int maxIter)
{
    const int N = (int)ptsL.size();
    Eigen::Matrix3d bestF = Eigen::Matrix3d::Identity();
    int bestInliers = 0;

    inlierMask.assign(N, false);

    // sampsonError() returns a SQUARED pixel distance (first-order Sampson
    // approximation), while `threshold` is specified as a linear pixel
    // tolerance — matching cv::findFundamentalMat's ransacReprojThreshold
    // convention so custom vs. OpenCV stay comparable at any threshold value.
    const double thresholdSq = threshold * threshold;

    // Setup random number generator
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);

    // dynamically calculate stopping criterion.
    int dynamicMaxIter = maxIter;

    // RANSAC Sampling Loop
    for (int it = 0; it < maxIter; ++it)
    {
        std::vector<int> idx;
        // Randomly select 8 unique point pair indexes
        while ((int)idx.size() < 8)
        {
            int r = dist(rng);
            if (std::find(idx.begin(), idx.end(), r) == idx.end())
            {
                idx.push_back(r);
            }
        }

        std::vector<cv::Point2f> sL(8), sR(8);
        for (int i = 0; i < 8; ++i)
        {
            sL[i] = ptsL[idx[i]];
            sR[i] = ptsR[idx[i]];
        }

        // matrix hypothesis
        Eigen::Matrix3d F = compute8Point(sL, sR);
        std::vector<bool> mask(N);
        int inliers = 0;

        // Evaluate model fit quality over the whole population using Sampson distance
        for (int i = 0; i < N; ++i)
        {
            mask[i] = sampsonError(F, ptsL[i], ptsR[i]) < thresholdSq;
            if (mask[i])
                ++inliers;
        }

        // Keep model track records if consensus exceeds previous records
        if (inliers > bestInliers)
        {
            bestInliers = inliers;
            bestF = F;
            inlierMask = mask;

            int iter_needed = calculateRequiredIterations(bestInliers, N, 8, confidence);
            dynamicMaxIter = std::min(dynamicMaxIter, iter_needed);
        }

        if (it >= dynamicMaxIter)
        {
            std::cout << "[RANSAC] Early termination triggered at iteration " << it
                      << " (Inliers: " << bestInliers << "/" << N << ")\n";
            break;
        }
    }

    // Recompute F over the entire collected inlier set to minimize noise drift
    std::vector<cv::Point2f> inL, inR;
    for (int i = 0; i < N; ++i)
    {
        if (inlierMask[i])
        {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
        }
    }

    // Recompute if the global inlier pool satisfies basic system dimensions
    if (inL.size() >= 8)
    {
        bestF = compute8Point(inL, inR);
        // Refresh final outlier rejection tracking arrays
        for (int i = 0; i < N; ++i)
        {
            inlierMask[i] = sampsonError(bestF, ptsL[i], ptsR[i]) < thresholdSq;
        }
    }
    return bestF;
}

Eigen::Matrix3d FundamentalMatrix::computeOpenCVRANSAC(
    const std::vector<cv::Point2f> &ptsL,
    const std::vector<cv::Point2f> &ptsR,
    std::vector<bool> &inlierMask,
    double threshold,
    double confidence)
{
    cv::Mat cvInlierMask;
    // Call OpenCV's native RANSAC implementation
    cv::Mat F_cv = cv::findFundamentalMat(ptsL, ptsR, cv::FM_RANSAC, threshold, confidence, cvInlierMask);

    // Map internal status back to our boolean inlier mask format
    inlierMask.resize(ptsL.size());
    for (size_t i = 0; i < ptsL.size(); ++i)
    {
        inlierMask[i] = (cvInlierMask.at<uchar>(i) != 0);
    }

    Eigen::Matrix3d F_eigen = Eigen::Matrix3d::Identity();
    if (!F_cv.empty())
    {
        // Enforce conversion format safety bounds between OpenCV types and Eigen
        if (F_cv.type() == CV_64F)
        {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    F_eigen(r, c) = F_cv.at<double>(r, c);
        }
        else if (F_cv.type() == CV_32F)
        {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    F_eigen(r, c) = F_cv.at<float>(r, c);
        }
    }
    return F_eigen;
}

double FundamentalMatrix::magsacWeight(double e, double sigmaMax)
{
    const int steps = 50;
    double integral = 0.0;
    double dSigma = sigmaMax / steps;

    for (int k = 1; k <= steps; ++k) {
        double sigma = k * dSigma;
        double sigma2 = sigma * sigma;

        double probability = std::exp(-e / (2.0 * sigma2));

        integral += probability * dSigma;
    }

    return integral / sigmaMax;
}

Eigen::Matrix3d FundamentalMatrix::computeWeighted8Point(
    const std::vector<cv::Point2f> &ptsL,
    const std::vector<cv::Point2f> &ptsR,
    const std::vector<double> &weights)
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

        double w = std::sqrt(std::max(weights[i], 1e-12));

        A.row(i) << w * xr * xl,
            w * xr * yl,
            w * xr,
            w * yr * xl,
            w * yr * yl,
            w * yr,
            w * xl,
            w * yl,
            w;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd f = svd.matrixV().col(8);

    Eigen::Matrix3d F;
    F << f(0), f(1), f(2),
        f(3), f(4), f(5),
        f(6), f(7), f(8);

    Eigen::JacobiSVD<Eigen::Matrix3d> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s = svdF.singularValues();
    s(2) = 0.0;

    F = svdF.matrixU() * s.asDiagonal() * svdF.matrixV().transpose();

    Eigen::Matrix3d Fdenorm = Tr.transpose() * F * Tl;

    if (std::abs(Fdenorm(2, 2)) > 1e-10)
    {
        Fdenorm /= Fdenorm(2, 2);
    }

    return Fdenorm;
}

Eigen::Matrix3d FundamentalMatrix::computeCustomMAGSAC(
    const std::vector<cv::Point2f> &ptsL,
    const std::vector<cv::Point2f> &ptsR,
    std::vector<bool> &inlierMask,
    double sigmaMax,
    double confidence,
    int maxIter)
{
    const int N = (int)ptsL.size();

    Eigen::Matrix3d bestF = Eigen::Matrix3d::Identity();
    double bestScore = -1.0;
    int bestInliers = 0;
    int bestEffectiveInliers = 0;

    inlierMask.assign(N, false);

    if (ptsR.size() != ptsL.size() || N < 8 || sigmaMax <= 0.0 ||
        confidence <= 0.0 || confidence >= 1.0 || maxIter <= 0)
    {
        return bestF;
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);

    const int sampleSize = 8;
    const int minIterations = std::min(maxIter, 200);
    const double thresholdSq = sigmaMax * sigmaMax;
    int dynamicMaxIter = maxIter;

    for (int it = 0; it < maxIter; ++it)
    {

        std::vector<int> idx;

        while ((int)idx.size() < sampleSize)
        {
            int r = dist(rng);
            if (std::find(idx.begin(), idx.end(), r) == idx.end())
            {
                idx.push_back(r);
            }
        }

        std::vector<cv::Point2f> sL(sampleSize), sR(sampleSize);

        for (int i = 0; i < sampleSize; ++i)
        {
            sL[i] = ptsL[idx[i]];
            sR[i] = ptsR[idx[i]];
        }

        Eigen::Matrix3d F = compute8Point(sL, sR);

        double score = 0.0;
        int inliers = 0;

        for (int i = 0; i < N; ++i)
        {
            double e = sampsonError(F, ptsL[i], ptsR[i]);

            // MAGSAC-style marginalization over sigma
            double w = magsacWeight(e, sigmaMax);

            score += w;

            if (e < thresholdSq)
            {
                ++inliers;
            }
        }

        if (score > bestScore)
        {
            bestScore = score;
            bestF = F;
            bestInliers = inliers;
            bestEffectiveInliers = std::min(
                bestInliers, static_cast<int>(std::floor(bestScore)));

            // Approximate the soft consensus as an equivalent integer support.
            // Clamping it by the hard support prevents small positive weights
            // from distant outliers from making the stopping rule optimistic.
            int iterNeeded = calculateRequiredIterations(
                bestEffectiveInliers, N, sampleSize, confidence);
            dynamicMaxIter = std::min(
                maxIter, std::max(minIterations, iterNeeded));
        }

        if (it >= dynamicMaxIter)
        {
            std::cout << "[MAGSAC-inspired] Early termination triggered at iteration " << it
                      << " (Effective support: " << bestEffectiveInliers << "/" << N
                      << ", Inliers: " << bestInliers << "/" << N << ")\n";
            break;
        }
    }

    //Sigma-consensus refinement
    std::vector<double> weights(N, 0.0);

    for (int i = 0; i < N; ++i)
    {
        double e = sampsonError(bestF, ptsL[i], ptsR[i]);
        weights[i] = magsacWeight(e, sigmaMax);
    }
    

    std::vector<cv::Point2f> inL, inR;
    std::vector<double> inW;

    for (int i = 0; i < N; ++i)
    {
        if (weights[i] > 1e-6)
        {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
            inW.push_back(weights[i]);
        }
    }

    if (inL.size() >= 8)
    {
        bestF = computeWeighted8Point(inL, inR, inW);
    }

    inlierMask.assign(N, false);

    for (int i = 0; i < N; ++i)
    {
        double e = sampsonError(bestF, ptsL[i], ptsR[i]);

        if (e < thresholdSq)
        {
            inlierMask[i] = true;
        }
    }

    return bestF;
}

Eigen::Matrix3d FundamentalMatrix::computeCustomPROSAC(
    const std::vector<cv::Point2f> &ptsL,
    const std::vector<cv::Point2f> &ptsR,
    std::vector<bool> &inlierMask,
    double threshold,
    double confidence,
    int maxIter)
{
    const int N = (int)ptsL.size();
    const int sampleSize = 8;

    Eigen::Matrix3d bestF = Eigen::Matrix3d::Identity();
    int bestInliers = 0;

    inlierMask.assign(N, false);

    std::mt19937 rng(42);

    int thresholdSq = threshold * threshold;

    // dynamic iterator T_N (stopping criterion)
    int dynamicIter = maxIter;
    int pool_size = sampleSize;
    // T_n
    double schedule_expander = 1.0;

    // Calculate inital T_n = T_8
    for (int i = 0; i < sampleSize; ++i)
    {
        schedule_expander *= static_cast<double>(sampleSize - i) / static_cast<double>(N - i);
    }

    schedule_expander *= maxIter;

    for (int t = 1; t <= maxIter; ++t)
    {
        while (t > schedule_expander && pool_size < N)
        {
            schedule_expander *= static_cast<double>(pool_size + 1) / static_cast<double>(pool_size + 1 - sampleSize);
            pool_size++;
        }

        std::vector<int> idx;
        idx.reserve(sampleSize);

        // base case: m = n = 8
        if (pool_size == sampleSize)
        {
            for (int i = 0; i < sampleSize; ++i)
            {
                idx.push_back(i);
            }
        }
        else
        {
            // constraint sampling
            // force to draw new sample n+1
            idx.push_back(pool_size - 1);

            // draw the remaining 7 points uniformly from the older points
            std::uniform_int_distribution<int> dist(0, pool_size - 2);

            while ((int)idx.size() < sampleSize)
            {
                int r = dist(rng);

                // Ensure we don't pick duplicates
                if (std::find(idx.begin(), idx.end(), r) == idx.end())
                {
                    idx.push_back(r);
                }
            }
        }

        std::vector<cv::Point2f> sL(sampleSize), sR(sampleSize);

        for (int i = 0; i < sampleSize; ++i)
        {
            sL[i] = ptsL[idx[i]];
            sR[i] = ptsR[idx[i]];
        }

        Eigen::Matrix3d F = compute8Point(sL, sR);

        std::vector<bool> mask(N, false);
        int inliers_total = 0;
        int inliers_pool = 0;

        // IMPORTANT:
        // evaluate on ALL matches, not only poolSize matches
        for (int i = 0; i < N; ++i)
        {
            double e = sampsonError(F, ptsL[i], ptsR[i]);

            if (e < thresholdSq)
            {
                mask[i] = true;
                ++inliers_total;

                // Assuming ptsL/ptsR are sorted by quality, check if it's in the current pool
                if (i < pool_size)
                {
                    ++inliers_pool;
                }
            }
        }

        if (inliers_total > bestInliers)
        {
            bestInliers = inliers_total;
            bestF = F;
            inlierMask = mask;
        }

        // non-randomness check

        const double beta = 0.05;
        int n_prime = pool_size - sampleSize;    // Trials
        int i_prime = inliers_pool - sampleSize; // Successes

        bool non_random = false;

        if (n_prime > 0)
        {
            // normal approximation of the binomial distribution
            double mu = n_prime * beta;
            double sigma = std::sqrt(n_prime * beta * (1.0 - beta));

            // if our successes exceed the expected random noise:
            if (i_prime > mu + 1.645 * sigma)
            {
                non_random = true;
            }
        }

        if (non_random)
        {
            // maximality constraint
            int iter_needed = calculateRequiredIterations(bestInliers, N, sampleSize, confidence);
            dynamicIter = std::min(maxIter, iter_needed);
        }

        // stopping criterion
        if (t >= dynamicIter)
        {
            std::cout << "[PROSAC] Early termination triggered at iteration " << t
                      << " (Inliers: " << bestInliers << "/" << N << ")\n";
            break;
        }
    }

    // Refit using all geometric inliers
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
        bestF = compute8Point(inL, inR);

        for (int i = 0; i < N; ++i)
        {
            inlierMask[i] = sampsonError(bestF, ptsL[i], ptsR[i]) < thresholdSq;
        }
    }

    return bestF;
}

int FundamentalMatrix::calculateRequiredIterations(int bestInliers, int N, int sampleSize, double confidence)
{
    if (N == 0)
        return std::numeric_limits<int>::max();

    double w = (double)bestInliers / (double)N;
    double p_fail = 1.0 - std::pow(w, sampleSize);

    // Prevent log(0) bounds
    p_fail = std::max(std::numeric_limits<double>::epsilon(), p_fail);
    p_fail = std::min(1.0 - std::numeric_limits<double>::epsilon(), p_fail);

    double log_prob = std::log(1.0 - confidence);
    double log_fail = std::log(p_fail);

    return (int)(log_prob / log_fail);
}
