#include <opencv2/calib3d.hpp>
#include <eigen3/Eigen/Dense>
#include "Rectification.hpp"

// Convert Eigen 3x3 to cv::Mat (CV_64F)
cv::Mat Rectification::toCvMat(const Eigen::Matrix3d& M)
{
    cv::Mat out(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out.at<double>(i, j) = M(i, j);
    return out;
}

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