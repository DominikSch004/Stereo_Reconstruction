#include <opencv2/calib3d.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "Rectification.hpp"

// OpenCV backend: calibrated rectification via stereoRectify + remap.
static bool computeCalibratedOpenCV(
    const cv::Mat& K,
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Size& imageSize,
    const cv::Mat& grayL,
    const cv::Mat& grayR,
    const cv::Mat& colorL,
    RectifyResult& out)
{
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    cv::stereoRectify(K, dist, K, dist, imageSize, R, t,
                      out.R1, out.R2, out.P1, out.P2, out.Q,
                      cv::CALIB_ZERO_DISPARITY, -1);

    cv::Mat mapAx, mapAy, mapBx, mapBy;
    cv::initUndistortRectifyMap(K, dist, out.R1, out.P1, imageSize, CV_16SC2, mapAx, mapAy);
    cv::initUndistortRectifyMap(K, dist, out.R2, out.P2, imageSize, CV_16SC2, mapBx, mapBy);
    cv::remap(grayL,  out.rectLeft,  mapAx, mapAy, cv::INTER_LINEAR);
    cv::remap(grayR,  out.rectRight, mapBx, mapBy, cv::INTER_LINEAR);
    cv::remap(colorL, out.rectColor, mapAx, mapAy, cv::INTER_LINEAR);
    return true;
}

bool Rectification::computeCalibrated(
    const cv::Mat& K,
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Size& imageSize,
    const cv::Mat& grayL,
    const cv::Mat& grayR,
    const cv::Mat& colorL,
    RectifyResult& out,
    RectificationMethod method)
{
    switch (method) {
        case RectificationMethod::Manual:
            std::cerr << "ERROR: manual rectification is not implemented yet.\n";
            return false;
        case RectificationMethod::OpenCV:
        default:
            return computeCalibratedOpenCV(K, R, t, imageSize, grayL, grayR, colorL, out);
    }
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