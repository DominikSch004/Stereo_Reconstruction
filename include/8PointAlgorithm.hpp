#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

class EightPointAlgorithm
{
public:
    // Hartley normalization: zero mean, average distance sqrt(2)
    // Returns T such that p_norm = T * p (homogeneous)
    static Eigen::Matrix3d normalizePoints(
        const std::vector<cv::Point2f>& pts,
        std::vector<cv::Point2f>& ptsNorm);

// Normalized 8-point algorithm → Fundamental matrix
    static Eigen::Matrix3d compute(
        const std::vector<cv::Point2f>& ptsL,
        const std::vector<cv::Point2f>& ptsR);
};