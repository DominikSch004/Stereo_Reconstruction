#include "FundamentalMatrix.hpp"
#include <opencv2/calib3d.hpp>
#include <Eigen/SVD>
#include <random>
#include <algorithm>
#include <cmath>

double FundamentalMatrix::sampsonError(
    const Eigen::Matrix3d& F,
    const cv::Point2f& pl,
    const cv::Point2f& pr)
{
    // Convert 2D pixel coordinates to homogeneous 3D vectors
    Eigen::Vector3d l(pl.x, pl.y, 1.0);
    Eigen::Vector3d r(pr.x, pr.y, 1.0);

    // Compute epipolar lines
    Eigen::Vector3d Fl  = F * l;
    Eigen::Vector3d Ftr = F.transpose() * r;

    // Algebraic error: r^T * F * l
    double num = r.dot(Fl);
    
    // sum of squared entry gradients for both images (denominator of the Sampson distance)
    double den = Fl(0)*Fl(0) + Fl(1)*Fl(1) +
                 Ftr(0)*Ftr(0) + Ftr(1)*Ftr(1);

    // first-order geometric approximation of the epipolar distance
    return (num * num) / den;
}

Eigen::Matrix3d FundamentalMatrix::normalizePoints(
    const std::vector<cv::Point2f>& pts,
    std::vector<cv::Point2f>& ptsNorm)
{
    // centroid (mean x, y) of the 2D point cloud
    double cx = 0, cy = 0;
    for (const auto& p : pts) {
        cx += p.x;
        cy += p.y;
    }
    cx /= pts.size();
    cy /= pts.size();

    // average distance from the shifting center point
    double scale = 0;
    for (const auto& p : pts) {
        scale += std::sqrt((p.x - cx)*(p.x - cx) + (p.y - cy)*(p.y - cy));
    }
    
    // Scale factor sets average distance of transformed points to sqrt(2)
    scale = std::sqrt(2.0) * pts.size() / scale;

    // Shift points to centroid and scale
    ptsNorm.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        ptsNorm[i] = cv::Point2f(
            float((pts[i].x - cx) * scale),
            float((pts[i].y - cy) * scale)
        );
    }

    // 3x3 transformation matrix T: p_norm = T * p
    Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
    T(0,0) = scale; T(1,1) = scale;
    T(0,2) = -cx * scale; T(1,2) = -cy * scale;
    T(2,2) = 1.0;

    return T;
}

Eigen::Matrix3d FundamentalMatrix::compute8Point(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR)
{
    // Hartley normalization to ensure numerical stability
    std::vector<cv::Point2f> Ln, Rn;
    Eigen::Matrix3d Tl = normalizePoints(ptsL, Ln);
    Eigen::Matrix3d Tr = normalizePoints(ptsR, Rn);

    // homogeneous linear system matrix A from epipolar constraint equations
    const int n = (int)Ln.size();
    Eigen::MatrixXd A(n, 9);

    for (int i = 0; i < n; ++i) {
        double xl = Ln[i].x, yl = Ln[i].y;
        double xr = Rn[i].x, yr = Rn[i].y;
        A.row(i) << xr*xl, xr*yl, xr, yr*xl, yr*yl, yr, xl, yl, 1.0;
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
    if (std::abs(Fdenorm(2,2)) > 1e-10) {
        Fdenorm /= Fdenorm(2,2);
    }
    return Fdenorm;
}

Eigen::Matrix3d FundamentalMatrix::computeFundamental(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    std::vector<bool>& inlierMask,
    FundamentalMethod method,
    double threshold,
    double confidence,
    int maxIter)
{
    if (method == FundamentalMethod::CustomRANSAC)
        return computeCustomRANSAC(ptsL, ptsR, inlierMask, threshold, maxIter);
    else if (method == FundamentalMethod::OpenCVRANSAC)
        return computeOpenCVRANSAC(ptsL, ptsR, inlierMask, threshold, confidence);
    else
        return computeCustomMAGSAC(ptsL, ptsR, inlierMask, threshold, maxIter);
}

Eigen::Matrix3d FundamentalMatrix::computeCustomRANSAC(
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

    // sampsonError() returns a SQUARED pixel distance (first-order Sampson
    // approximation), while `threshold` is specified as a linear pixel
    // tolerance — matching cv::findFundamentalMat's ransacReprojThreshold
    // convention so custom vs. OpenCV stay comparable at any threshold value.
    const double thresholdSq = threshold * threshold;

    // Setup random number generator
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);

    // RANSAC Sampling Loop
    for (int it = 0; it < maxIter; ++it) {
        std::vector<int> idx;
        // Randomly select 8 unique point pair indexes
        while ((int)idx.size() < 8) { 
            int r = dist(rng);
            if (std::find(idx.begin(), idx.end(), r) == idx.end()) {
                idx.push_back(r);
            }
        }

        std::vector<cv::Point2f> sL(8), sR(8);
        for (int i = 0; i < 8; ++i) {
            sL[i] = ptsL[idx[i]];
            sR[i] = ptsR[idx[i]];
        }

        // matrix hypothesis
        Eigen::Matrix3d F = compute8Point(sL, sR);
        std::vector<bool> mask(N);
        int inliers = 0;

        // Evaluate model fit quality over the whole population using Sampson distance
        for (int i = 0; i < N; ++i) {
            mask[i] = sampsonError(F, ptsL[i], ptsR[i]) < thresholdSq;
            if (mask[i]) ++inliers;
        }

        // Keep model track records if consensus exceeds previous records
        if (inliers > bestInliers) {
            bestInliers = inliers;
            bestF = F;
            inlierMask = mask;
        }
    }

    // Recompute F over the entire collected inlier set to minimize noise drift
    std::vector<cv::Point2f> inL, inR;
    for (int i = 0; i < N; ++i) {
        if (inlierMask[i]) {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
        }
    }

    // Recompute if the global inlier pool satisfies basic system dimensions
    if (inL.size() >= 8) {
        bestF = compute8Point(inL, inR);
        // Refresh final outlier rejection tracking arrays
        for (int i = 0; i < N; ++i) {
            inlierMask[i] = sampsonError(bestF, ptsL[i], ptsR[i]) < thresholdSq;
        }
    }
    return bestF;
}

Eigen::Matrix3d FundamentalMatrix::computeOpenCVRANSAC(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    std::vector<bool>& inlierMask,
    double threshold,
    double confidence)
{
    cv::Mat cvInlierMask;
    // Call OpenCV's native RANSAC implementation
    cv::Mat F_cv = cv::findFundamentalMat(ptsL, ptsR, cv::FM_RANSAC, threshold, confidence, cvInlierMask);

    // Map internal status back to our boolean inlier mask format
    inlierMask.resize(ptsL.size());
    for (size_t i = 0; i < ptsL.size(); ++i) {
        inlierMask[i] = (cvInlierMask.at<uchar>(i) != 0);
    }

    Eigen::Matrix3d F_eigen = Eigen::Matrix3d::Identity();
    if (!F_cv.empty()) {
        // Enforce conversion format safety bounds between OpenCV types and Eigen
        if (F_cv.type() == CV_64F) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    F_eigen(r, c) = F_cv.at<double>(r, c);
        } else if (F_cv.type() == CV_32F) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    F_eigen(r, c) = F_cv.at<float>(r, c);
        }
    }
    return F_eigen;
}





Eigen::Matrix3d FundamentalMatrix::computeWeighted8Point(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    const std::vector<double>& weights)
{
    std::vector<cv::Point2f> Ln, Rn;
    Eigen::Matrix3d Tl = normalizePoints(ptsL, Ln);
    Eigen::Matrix3d Tr = normalizePoints(ptsR, Rn);

    const int n = (int)Ln.size();
    Eigen::MatrixXd A(n, 9);

    for (int i = 0; i < n; ++i) {
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

    if (std::abs(Fdenorm(2,2)) > 1e-10) {
        Fdenorm /= Fdenorm(2,2);
    }

    return Fdenorm;
}




Eigen::Matrix3d FundamentalMatrix::computeCustomMAGSAC(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    std::vector<bool>& inlierMask,
    double sigmaMax,
    int maxIter)
{
    const int N = (int)ptsL.size();

    Eigen::Matrix3d bestF = Eigen::Matrix3d::Identity();
    double bestScore = -1.0;

    inlierMask.assign(N, false);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);

    const int sampleSize = 8;

    for (int it = 0; it < maxIter; ++it) {

        std::vector<int> idx;

        while ((int)idx.size() < sampleSize) {
            int r = dist(rng);
            if (std::find(idx.begin(), idx.end(), r) == idx.end()) {
                idx.push_back(r);
            }
        }

        std::vector<cv::Point2f> sL(sampleSize), sR(sampleSize);

        for (int i = 0; i < sampleSize; ++i) {
            sL[i] = ptsL[idx[i]];
            sR[i] = ptsR[idx[i]];
        }

        Eigen::Matrix3d F = compute8Point(sL, sR);

        double score = 0.0;

        for (int i = 0; i < N; ++i) {
            double e = sampsonError(F, ptsL[i], ptsR[i]);

            // MAGSAC-style soft truncated Gaussian score
            if (e < sigmaMax * sigmaMax) {
                double w = std::exp(-e / (2.0 * sigmaMax * sigmaMax));
                score += w;
            }
        }

        if (score > bestScore) {
            bestScore = score;
            bestF = F;
        }
    }

    std::vector<double> weights(N, 0.0);

    for (int i = 0; i < N; ++i) {
        double e = sampsonError(bestF, ptsL[i], ptsR[i]);

        if (e < sigmaMax * sigmaMax) {
            weights[i] = std::exp(-e / (2.0 * sigmaMax * sigmaMax));
        } else {
            weights[i] = 0.0;
        }
    }

    std::vector<cv::Point2f> inL, inR;
    std::vector<double> inW;

    for (int i = 0; i < N; ++i) {
        if (weights[i] > 1e-6) {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
            inW.push_back(weights[i]);
        }
    }

    if (inL.size() >= 8) {
        bestF = computeWeighted8Point(inL, inR, inW);
    }

    inlierMask.assign(N, false);

    for (int i = 0; i < N; ++i) {
        double e = sampsonError(bestF, ptsL[i], ptsR[i]);

        if (e < sigmaMax * sigmaMax) {
            inlierMask[i] = true;
        }
    }

    return bestF;
}