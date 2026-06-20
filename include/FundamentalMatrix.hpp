#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

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
};