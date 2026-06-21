#pragma once

#include <opencv2/core.hpp>

/**
 * @enum DisparityMethod
 * @brief Selects the cost volume metric for hand-rolled dense block-matching.
 * SAD: Sum of Absolute Differences, SSD: Sum of Squared Differences, NCC: Normalized Cross-Correlation, OpenCVSGBM: OpenCV's optimized Semi-Global Block Matching (SGBM).
 */
enum class DisparityMethod {
    SSD,        // Sum of Squared Differences
    SAD,        // Sum of Absolute Differences
    NCC,        // Normalized Cross-Correlation
    OpenCVSGBM // OpenCV's Semi-Global Block Matching (SGBM)
};

/**
 * @class Disparity
 * @brief Handles dense stereo matching cost volumes and disparity map generation.
 */
class Disparity
{
public:
    /**
     * @brief High-level entrypoint that computes a dense disparity map using the requested method.
     * @param left Rectified grayscale left image.
     * @param right Rectified grayscale right image.
     * @param minDisp Minimum disparity search bound.
     * @param numDisp Total number of disparity levels to search.
     * @param blockSize Odd window size diameter for pixel aggregation.
     * @param method The cost metric or backend engine choice.
     * @return CV_32F disparity map matrix containing actual pixel disparities.
     */
    static cv::Mat computeDisparity(
        const cv::Mat& left,
        const cv::Mat& right,
        int minDisp,
        int numDisp,
        int blockSize,
        DisparityMethod method = DisparityMethod::OpenCVSGBM
    );

private:
    /**
     * @brief Internal helper functions for each cost metric. Each computes a raw cost volume and selects the best disparity per pixel.
     */
    static cv::Mat computeSSD(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize);
    static cv::Mat computeSAD(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize);
    static cv::Mat computeNCC(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize);
    static cv::Mat computeSGBMOpenCV(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize);
};