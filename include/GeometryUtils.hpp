#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

namespace GeometryUtils
{

    // extracts 3D Pose from a 2D Fundamental Matrix
    bool extractPoseFromFundamental(const Eigen::Matrix3d &F_eigen,
                                    const std::vector<cv::Point2f> &ptsL,
                                    const std::vector<cv::Point2f> &ptsR,
                                    const std::vector<bool> &inlierMask,
                                    const cv::Mat &K,
                                    Eigen::Matrix3d &R_est,
                                    Eigen::Vector3d &t_est);

}