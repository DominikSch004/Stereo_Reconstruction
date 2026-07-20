#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

// Rescales a pixel-unit/area threshold given at full resolution (scale=1.0) to processingScale.
int scaleLinear(double base, double scale, int minVal);
int scaleArea(double base, double scale, int minVal);

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
 * @struct DisparityRefinementConfig
 * @brief Custom backend only. Toggles and base (scale=1.0) thresholds for
 * Hirschmuller 2008 Sec 2.5, "Disparity Refinement": postprocessing stages
 * layered on top of the core disparity pipeline (Sec 2.1-2.3, Fig. 3, which
 * always runs in full -- subpixel estimation and the L/R consistency check
 * are part of that core and are not configurable). Peak filtering (Sec
 * 2.5.1) and gap interpolation (Sec 2.5.3) can each be disabled independently
 * to inspect its individual effect. minPeakSegmentPx is given at full
 * resolution and rescaled internally by config.processing_scale, mirroring
 * cv::StereoSGBM's speckleWindowSize. The segment-membership tolerance itself
 * (max 1px disparity step between neighbors) is fixed by the paper's own
 * definition of a peak, not a free parameter -- see removePeaks.
 */
struct DisparityRefinementConfig
{
    bool peakFiltering = true;  // Sec 2.5.1: drop small isolated disparity segments ("peaks")
    int minPeakSegmentPx = 100; // min connected-segment area (px^2) to survive peak filtering;
                                 // not paper-specified ("a certain size"), mirrors cv::StereoSGBM's speckleWindowSize
    bool gapFill = false;       // Sec 2.5.3: interpolate remaining invalid pixels to reach full coverage
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
     * @param refinement Custom backend only: see DisparityRefinementConfig.
     * @return CV_32F disparity map matrix containing actual pixel disparities.
     */
    static cv::Mat computeDisparity(
        const cv::Mat &left,
        const cv::Mat &right,
        int minDisp,
        int numDisp,
        int blockSize,
        DisparityMethod method,
        double scale = 0.5,
        const DisparityRefinementConfig &refinement = DisparityRefinementConfig());

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
    static cv::Mat computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, double scale, const DisparityRefinementConfig &refinement);

    // Custom SGM helpers (Birchfield-Tomasi cost + 16-direction SGM)
    static std::vector<uint16_t> computePixelwiseCost(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, bool rightBase = false);
    static void computeBTIntervals(const cv::Mat &src, cv::Mat &Imin, cv::Mat &Imax);
    static void aggregatePathCost(const std::vector<uint16_t> &C, std::vector<uint16_t> &S, int rows, int cols, int numDisp, int dx, int dy, int P1, int P2);
    static cv::Mat computeWTADisparity(const cv::Mat &left, const cv::Mat &right, int rows, int cols, int minDisp, int numDisp, int P1, int P2, bool rightBase);
    static float estimateSubpixel(const uint16_t *costsAtPixel, int numDisp, int bestDispIdx);
    static cv::Mat interpolateGaps(const cv::Mat &disparity, const cv::Mat &dispRight, const cv::Mat &baseF, int minDisp, int numDisp);
    static cv::Mat nearestValidInDirection(const cv::Mat &disp, int minDisp, int rows, int cols, int dx, int dy);
    static cv::Mat removePeaks(const cv::Mat &disparity, int minDisp, int minSegmentSize, float maxSegmentDispDiff = 1.0f);
};

