#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace VisualizationUtils
{

    // Draws epipolar lines and matching points then displays them
    void displayEpipolarMatches(const std::string &windowTitle,
                                const cv::Mat &imgL,
                                const cv::Mat &imgR,
                                const std::vector<cv::Point2f> &ptsL,
                                const std::vector<cv::Point2f> &ptsR,
                                const std::vector<bool> &inlierMask,
                                const Eigen::Matrix3d &F,
                                int maxDrawn = 20);
}