#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

// Selects how the calibrated stereo rectification is computed.
enum class RectificationMethod {
    OpenCV,   // cv::stereoRectify + cv::initUndistortRectifyMap / cv::remap
    Manual    // hand-rolled rectification (not implemented yet)
};

// Outputs of a calibrated rectification.
struct RectifyResult {
    cv::Mat R1, R2;                          // rectifying rotations (left, right)
    cv::Mat P1, P2;                          // rectified 3x4 projection matrices
    cv::Mat Q;                               // 4x4 disparity-to-depth matrix
    cv::Mat rectLeft, rectRight, rectColor;  // rectified images
};

class Rectification
{
public:
    // Calibrated rectification used by the main pipeline. Given the intrinsics K
    // and the relative pose (R, t) between the two cameras, fills out with the
    // rectifying transforms, Q, and the rectified images. Returns false on failure.
    static bool computeCalibrated(
        const cv::Mat& K,
        const cv::Mat& R,
        const cv::Mat& t,
        const cv::Size& imageSize,
        const cv::Mat& grayL,
        const cv::Mat& grayR,
        const cv::Mat& colorL,
        RectifyResult& out,
        RectificationMethod method = RectificationMethod::OpenCV
    );

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