#include <opencv2/calib3d.hpp>
#include <eigen3/Eigen/Dense>
#include "Rectification.hpp"

bool Rectification::computeUncalibrated(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    const cv::Size& imageSize,
    const cv::Mat& F,
    cv::Mat& H1,
    cv::Mat& H2)
{
    if (ptsL.size() < 8) return false;

    return cv::stereoRectifyUncalibrated(
        ptsL,
        ptsR,
        F,
        imageSize,
        H1,
        H2
    );
}

void Rectification::warp(
    const cv::Mat& imgL,
    const cv::Mat& imgR,
    const cv::Mat& H1,
    const cv::Mat& H2,
    cv::Mat& rectL,
    cv::Mat& rectR)
{
    cv::warpPerspective(imgL, rectL, H1, imgL.size());
    cv::warpPerspective(imgR, rectR, H2, imgR.size());
}