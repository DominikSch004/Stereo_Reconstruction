#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

/**
 * @enum RectificationMethod
 * @brief Selects the backend engine used for calibrated stereo rectification.
 */
enum class RectificationMethod {
    CalibratedOpenCV, // Native OpenCV stereoRectify + remap pipeline
    CalibratedCustom  // Custom hand-rolled projective rectification (Loop and Zhang)
};

/**
 * @struct RectifyResult
 * @brief Outputs of a calibrated rectification.
 */
struct RectifyResult {
    cv::Mat R1, R2;                          // rectifying rotations (left, right) to virtually rotate cameras into a common plane
    cv::Mat P1, P2;                          // rectified 3x4 projection matrices mapping 3D points directly to the rectified 2D pixel coordinates
    cv::Mat Q;                               // 4x4 disparity-to-depth matrix, transforms a pixel coordinate (x, y) and disparity d into a 3D point (X, Y, Z) 
    cv::Mat rectLeft, rectRight, rectColor;  // rectified images
};

/**
 * @class Rectification
 * @brief Handles geometric image warping to align epipolar lines horizontally.
 */
class Rectification
{
public:
    /**
     * @brief High-level unified entrypoint for calibrated stereo rectification.
     * @param K Camera intrinsic calibration matrix (assumed shared).
     * @param R Relative rotation matrix from Left to Right camera frame.
     * @param t Relative translation vector from Left to Right camera frame.
     * @param imageSize Dimensions of the input frames.
     * @param grayL Grayscale left frame.
     * @param grayR Grayscale right frame.
     * @param colorL Color left frame (used to extract high-res textured PLY point clouds).
     * @param[out] out Result data bag populated with matrices and row-aligned frames.
     * @param method Selects the execution backend (CalibratedOpenCV or CalibratedCustom).
     * @return true if mapping configurations generation and warps succeed, false otherwise.
     */
    static bool computeCalibrated(
        const cv::Mat& K,
        const cv::Mat& R,
        const cv::Mat& t,
        const cv::Size& imageSize,
        const cv::Mat& grayL,
        const cv::Mat& grayR,
        const cv::Mat& colorL,
        RectifyResult& out,
        RectificationMethod method = RectificationMethod::CalibratedOpenCV
    );

    /**
     * @brief Uncalibrated rectification via foundational homographies.
     * @details Computes rectification homographies (H1, H2) using only matched keypoints and F.
     */
    static bool computeUncalibrated(
        const std::vector<cv::Point2f>& ptsL,
        const std::vector<cv::Point2f>& ptsR,
        const cv::Size& imageSize,
        const cv::Mat& F,
        cv::Mat& H1,
        cv::Mat& H2
    );

    /**
     * @brief Warps target images using 3x3 projective homography transformations.
     */
    static void warp(
        const cv::Mat& imgL,
        const cv::Mat& imgR,
        const cv::Mat& H1,
        const cv::Mat& H2,
        cv::Mat& rectL,
        cv::Mat& rectR
    );

private:
    /**
     * @brief Calibrated rectification via OpenCV baseline (stereoRectify + remap).
     * @details Given the intrinsics K and the relative pose (R, t) between the two cameras, 
     * fills out with the rectifying transforms, Q, and the rectified images. Returns false on failure.
     */
    static bool computeCalibratedOpenCV(
        const cv::Mat& K,
        const cv::Mat& R,
        const cv::Mat& t,
        const cv::Size& imageSize,
        const cv::Mat& grayL,
        const cv::Mat& grayR,
        const cv::Mat& colorL,
        RectifyResult& out
    );

    /**
     * @brief Custom calibrated rectification (Hand-rolled implementation).
     * @note To be implemented in Week 3 following Loop and Zhang.
     */
    static bool computeCalibratedCustom(
        const cv::Mat& K,
        const cv::Mat& R,
        const cv::Mat& t,
        const cv::Size& imageSize,
        const cv::Mat& grayL,
        const cv::Mat& grayR,
        const cv::Mat& colorL,
        RectifyResult& out
    );
};