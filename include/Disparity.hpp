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
     * @param useIntensityConsistentSelection Custom backend only (Hirschmuller 2008 Sec 2.5.2)
     * @param useGapFill Custom backend only (Hirschmuller 2008 Sec 2.5.3)
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
        bool useIntensityConsistentSelection = false,
        bool useGapFill = false);

    static void computeDynamicSearchRangeCalibrated(
        const std::vector<cv::Point2f> &inPtsL,
        const std::vector<cv::Point2f> &inPtsR,
        const cv::Mat &K,
        const cv::Mat &R1, const cv::Mat &P1,
        const cv::Mat &R2, const cv::Mat &P2,
        const cv::Size &imgSize,
        int &minDisp,
        int &numDisp);

private:
    // OpenCV and custom SGM backend implementations
    static cv::Mat computeSGBMOpenCV(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, double scale);
    static cv::Mat computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, double scale,
                                 bool useIntensityConsistentSelection, bool useGapFill);

    // Custom SGM helpers (Birchfield-Tomasi cost + 16-direction SGM)
    static float btCost(float baseVal, float minBase, float maxBase, float matchVal, float minMatch, float maxMatch);
    static std::vector<uint16_t> computeCostVolume(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, bool rightBase = false);
    static void computeBTIntervals(const cv::Mat &src, cv::Mat &Imin, cv::Mat &Imax);
    static void aggregateDirection(const std::vector<uint16_t> &C, std::vector<uint16_t> &S, const cv::Mat &baseF, int rows, int cols, int numDisp, int dx, int dy, int P1, int P2Base);
    static cv::Mat computeWTADisparity(const cv::Mat &left, const cv::Mat &right, int rows, int cols, int minDisp, int numDisp, int P1, int P2Base, bool rightBase);
    static cv::Mat interpolateGaps(const cv::Mat &disparity, const cv::Mat &dispRight, int minDisp, int numDisp);
    static cv::Mat nearestValidInDirection(const cv::Mat &disp, int minDisp, int rows, int cols, int dx, int dy);
    static cv::Mat removePeaks(const cv::Mat &disparity, int minDisp, int minSegmentSize, float maxSegmentDispDiff = 1.0f);
    static cv::Mat meanShiftModes(const cv::Mat &baseF, int sigmaS, float sigmaR, int maxIters = 20, float convergeEps = 0.1f);
    static cv::Mat segmentByIntensity(const cv::Mat &modes, float mergeTolerance, int minSegmentSize);
    
    // A*x + B*y + C = disparity, fit by least squares over a pixel set.
    struct PlaneHypothesis
    {
        double a = 0.0, b = 0.0, c = 0.0;
        float at(int x, int y) const { return static_cast<float>(a * x + b * y + c); }
    };

    // Read-only inputs shared by findPlaneHypotheses/scoreHypothesis, bundled together
    // so each doesn't need its own long parameter list.
    struct SegmentEvalContext
    {
        const cv::Mat &labels, &disparity;
        const cv::Mat &baseF, &matchF;
        const cv::Mat &Imin_base, &Imax_base, &Imin_match, &Imax_match;
        int minDisp, maxDisp;
        float minDispF;
        int P1, P2Base;
        int rows, cols;
    };

    static bool fitPlane(const std::vector<cv::Point> &pixels, const cv::Mat &disparity, PlaneHypothesis &out);
    static std::vector<PlaneHypothesis> findPlaneHypotheses(const std::vector<cv::Point> &Si, int segLabel, const SegmentEvalContext &ctx);
    static double scoreHypothesis(const PlaneHypothesis &hyp, const std::vector<cv::Point> &Si, int segLabel, const SegmentEvalContext &ctx);
    static cv::Mat selectIntensityConsistentDisparity(const cv::Mat &disparity,
                                                       const cv::Mat &baseF, const cv::Mat &matchF,
                                                       const cv::Mat &Imin_base, const cv::Mat &Imax_base,
                                                       const cv::Mat &Imin_match, const cv::Mat &Imax_match,
                                                       int minDisp, int numDisp, int P1, int P2Base);
};
