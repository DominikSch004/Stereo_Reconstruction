#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <FundamentalMatrix.hpp>
#include "Rectification.hpp"

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
                                double rotErrorDeg,
                                double transErrorDeg,
                                double epipolarErrorPx,
                                int maxDrawn = 20);

    void visualizeOutliers(const std::vector<cv::Point2f> &ptsL,
                           const std::vector<cv::Point2f> &ptsR,
                           const cv::Mat &grayLeft,
                           const cv::Mat &grayRight,
                           const std::vector<bool> &inlierMask,
                           const std::string &leftWindowTitle,
                           const std::string &rightWindowTitle);

    void fundamentalExplorationVideo(
        const cv::Mat &imgL,
        const cv::Mat &imgR,
        const VisualizationData &visualize,
        const std::string &windowName,
        int pauseInterval = 0);

    void fundamentalComparison(
        const Eigen::Matrix3d &F,
        const Eigen::Matrix3d R_gt,
        const Eigen::Vector3d t_gt,
        const cv::Mat K);

    void visualizeRectification(
        const RectifyResult &rect,
        const std::vector<cv::Point2f> &inL,
        const std::vector<cv::Point2f> &inR,
        const cv::Mat &K,
        const std::string &windowName = "Rectification Verification");
}