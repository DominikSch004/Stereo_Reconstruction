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
    Custom      // Custom hand-rolled dense matcher (Birchfield-Tomasi cost + 16-direction SGM)
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
     * @param scale config.processingScale the images were downscaled by. Both backends
     * scale their own pixel/disparity-unit thresholds (L-R consistency tolerance, speckle/peak
     * region area and disparity-agreement range) by this factor so they stay comparable
     * across different scales.
     * @return CV_32F disparity map matrix containing actual pixel disparities.
     */
    static cv::Mat computeDisparity(
        const cv::Mat &left,
        const cv::Mat &right,
        int minDisp,
        int numDisp,
        int blockSize,
        DisparityMethod method,
        double scale = 0.5);

    static void computeDynamicSearchRangeCalibrated(
        const std::vector<cv::Point2f> &inPtsL,
        const std::vector<cv::Point2f> &inPtsR,
        const cv::Mat &K,
        const cv::Mat &R1, const cv::Mat &P1,
        const cv::Mat &R2, const cv::Mat &P2,
        const cv::Size &imgSize,
        int &minDisp,
        int &numDisp);

    /**
     * @brief Builds a soft stereo confidence map and invalidates pixels that fail
     *        a left/right consistency check.
     *
     * The returned CV_32F map combines cycle consistency and photometric
     * agreement in [0,1].  It is the c_stereo term used by confidence-weighted
     * ICP and is kept separate for the factor ablation.
     */
    static cv::Mat filterAndComputeConfidence(
        const cv::Mat &left,
        const cv::Mat &right,
        cv::Mat &disparityLeft,
        int minDisp,
        int numDisp,
        int blockSize,
        DisparityMethod method,
        float lrMaxDiff = 1.5f,
        float photometricScale = 25.0f);

private:
    // OpenCV and custom SGM backend implementations
    static cv::Mat computeSGBMOpenCV(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, double scale);
    static cv::Mat computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, double scale);

    // Custom SGM helpers (Birchfield-Tomasi cost + 16-direction SGM)
    static std::vector<uint16_t> computeCostVolume(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, bool rightBase = false);
    static void computeBTIntervals(const cv::Mat &src, cv::Mat &Imin, cv::Mat &Imax);
    static void aggregateDirection(const std::vector<uint16_t> &C, std::vector<uint16_t> &S, int rows, int cols, int numDisp, int dx, int dy, int P1, int P2);
    static cv::Mat computeWTADisparity(const cv::Mat &left, const cv::Mat &right, int rows, int cols, int minDisp, int numDisp, int P1, int P2, bool rightBase);
    static cv::Mat interpolateGaps(const cv::Mat &disparity, const cv::Mat &dispRight, int minDisp, int numDisp);
    static cv::Mat nearestValidInDirection(const cv::Mat &disp, int minDisp, int rows, int cols, int dx, int dy);
    static cv::Mat removePeaks(const cv::Mat &disparity, int minDisp, int minSegmentSize, float maxSegmentDispDiff = 1.0f);
};

