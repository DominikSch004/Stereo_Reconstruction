#pragma once

#include <opencv2/core.hpp>
#include "Types.hpp"

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
     * @return CV_32F disparity map matrix containing actual pixel disparities.
     */
    static cv::Mat computeDisparity(
        const cv::Mat &left,
        const cv::Mat &right,
        int minDisp,
        int numDisp,
        int blockSize,
        PipelineMode mode);

private:
    /**
     * @brief metric from OpenCV implementation
     */
    static cv::Mat computeSGBMOpenCV(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize);
    static cv::Mat computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize);
};