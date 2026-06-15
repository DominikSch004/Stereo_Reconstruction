#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

class Rectification
{
public:
    static bool computeUncalibrated(
        const std::vector<cv::Point2f>& ptsL,
        const std::vector<cv::Point2f>& ptsR,
        const cv::Size& imageSize,
        const cv::Mat& F,
        cv::Mat& H1,
        cv::Mat& H2
    );

    static void warp(
        const cv::Mat& imgL,
        const cv::Mat& imgR,
        const cv::Mat& H1,
        const cv::Mat& H2,
        cv::Mat& rectL,
        cv::Mat& rectR
    );
};