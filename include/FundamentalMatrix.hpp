#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

// Selects how the fundamental matrix is estimated from correspondences.
enum class FundamentalMethod {
    OpenCV,   // cv::findFundamentalMat (FM_RANSAC + 8-point)
    Manual    // hand-rolled RANSAC + normalized 8-point (EightPointAlgorithm)
};

class FundamentalMatrix
{
public:
    // Sampson distance (first-order approximation of epipolar error)
    static double sampsonError(
        const Eigen::Matrix3d& F,
        const cv::Point2f& pl,
        const cv::Point2f& pr);

    // RANSAC + 8-point: returns best F and fills inlierMask
    static Eigen::Matrix3d ransac(
        const std::vector<cv::Point2f>& ptsL,
        const std::vector<cv::Point2f>& ptsR,
        std::vector<bool>& inlierMask,
        double threshold = 1.0,
        int maxIter = 1000);

    // RANSAC + 8-point estimation of F, choosing the backend via method.
    // Returns F as a 3x3 CV_64F matrix and fills inlierMask with one entry per
    // input correspondence (non-zero = inlier). Drop-in replacement for the
    // output of cv::findFundamentalMat.
    static cv::Mat compute(
        const std::vector<cv::Point2f>& ptsL,
        const std::vector<cv::Point2f>& ptsR,
        std::vector<uchar>& inlierMask,
        FundamentalMethod method = FundamentalMethod::OpenCV,
        double threshold = 1.0,
        double confidence = 0.99,
        int maxIter = 1000);
};