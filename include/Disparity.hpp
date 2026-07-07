#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

/**
 * @enum DisparityMethod
 * @brief Selects the dense stereo matching backend.
 */
enum class DisparityMethod
{
    OpenCVSGBM, // Native OpenCV semi-global block matching
    Custom      // Custom hand-rolled dense matcher (not implemented yet)
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
     * @return CV_32F disparity map matrix containing actual pixel disparities.
     */
    static cv::Mat computeDisparity(
        const cv::Mat &left,
        const cv::Mat &right,
        int minDisp,
        int numDisp,
        int blockSize,
        DisparityMethod method);

private:
     // OpenCV and custom SGM backend implementations
    static cv::Mat computeSGBMOpenCV(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize);
    static cv::Mat computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize);

    // Custom SGM helpers (Birchfield-Tomasi cost volume)
    static std::vector<uint16_t> computeCostVolume(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, bool rightBase = false);
    static void computeBTIntervals(const cv::Mat &src, cv::Mat &Imin, cv::Mat &Imax);
    static void aggregateDirection(const std::vector<uint16_t> &C, std::vector<uint16_t> &S, int rows, int cols, int numDisp, int dx, int dy, int P1, int P2);
    static cv::Mat computeWTADisparity(const cv::Mat &left, const cv::Mat &right, int rows, int cols, int minDisp, int numDisp, int P1, int P2, bool rightBase);
    static cv::Mat interpolateGaps(const cv::Mat &disparity, const cv::Mat &dispRight, int minDisp, int numDisp);
    static cv::Mat nearestValidInDirection(const cv::Mat &disp, int minDisp, int rows, int cols, int dx, int dy);
};