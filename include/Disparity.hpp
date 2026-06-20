#pragma once

#include <opencv2/core.hpp>

/**
 * @enum DisparityMethod
 * @brief Selects the cost volume metric for hand-rolled dense block-matching.
 */
enum class DisparityMethod {
    SSD,  // Sum of Squared Differences
    SAD,  // Sum of Absolute Differences
    NCC   // Normalized Cross-Correlation
};

/**
 * @class Disparity
 * @brief Handles dense stereo matching cost volumes and disparity map generation.
 */
class Disparity
{
public:
    /**
     * @brief Custom dense block-matching pipeline using selectable cost metrics.
     * @details Computes a dense disparity map by sliding a window along horizontal epipolar lines.
     * @param left Rectified grayscale left image.
     * @param right Rectified grayscale right image.
     * @param minDisp Minimum disparity search bound (usually 0 or negative depending on setup).
     * @param numDisp Total number of disparity levels to search (must be divisible by 16 for hardware alignment).
     * @param blockSize Odd window size diameter (e.g., 3, 5, 7) for pixel aggregation.
     * @param method The cost metric to use (SAD, SSD, or NCC).
     * @return CV_32F disparity map matrix.
     */
    static cv::Mat computeCustom(
        const cv::Mat& left,
        const cv::Mat& right,
        int minDisp,
        int numDisp,
        int blockSize,
        DisparityMethod method = DisparityMethod::SAD);

    /**
     * @brief OpenCV baseline dense matcher using Semi-Global Block Matching (SGBM) with openCV
     * @details Acts as the gold-standard baseline for cost-volume evaluation.
     * @return CV_32F disparity map scaled down to actual pixel disparity values.
     */
    static cv::Mat computeSGBMOpenCV(
        const cv::Mat& left,
        const cv::Mat& right,
        int minDisp,
        int numDisp,
        int blockSize);

private:
    /**
     * @brief Internal helper functions for each cost metric. Each computes a raw cost volume and selects the best disparity per pixel.
     */
    static cv::Mat computeSSD(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize);
    static cv::Mat computeSAD(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize);
    static cv::Mat computeNCC(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize);
};