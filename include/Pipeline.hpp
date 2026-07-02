#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "DTULoader.hpp"
#include "Triangulation.hpp"
#include "Disparity.hpp"
#include "Types.hpp"

/**
 * @struct PipelineResult
 * @brief Self-contained data bag tracking the evaluation states of a single stereo view pair.
 */
struct PipelineResult
{
    cv::Mat rectLeft, rectRight, rectColor;
    cv::Mat Q;          // 4x4 disparity-to-depth matrix
    cv::Mat R1, R2;     // 3x3 rectifying rotations (left, right)
    cv::Mat P1r, P2r;   // 3x4 rectified projection matrices (left, right)
    cv::Mat K;          // intrinsics actually used (scaled to processing resolution)
    cv::Mat E;          // Store Essential Matrix calculation (for evaluator)
    cv::Mat R_est;      // (for evaluator)
    cv::Mat t_est;      // (for evaluator)
    cv::Mat camToWorld; // 3x4 [R|t]: camera coordinates -> world frame
    int minDisp = 0;    // Dynamic search range start
    int numDisp = 16;   // Dynamic search range width (multiple of 16)
    std::vector<cv::Point2f> inPtsL, inPtsR;
    std::vector<bool> inlierMask;
    cv::Size imgSize;
    cv::Mat denseDisparity; // CV_32F calculated dense correspondence map
    cv::Mat dense3DPoints;  // CV_32FC3 spatial point grid for downstream ICP pipelines
};

/**
 * @class Pipeline
 * @brief Orchestration engine coordinating feature matching, epipolar geometry,
 * stereo rectification, and dense matching search-bound configuration.
 */
class Pipeline
{
public:
    /**
     * @brief High-level entrypoint routing the execution flow based on the chosen mode.
     * @return true if the entire orchestration sequence finishes successfully, false otherwise.
     */
    static bool runPipeline(
        const cv::Mat &imgLeft,
        const cv::Mat &imgRight,
        const cv::Mat &K,
        PipelineResult &res,
        PipelineMode mode = PipelineMode::OpenCV);

private:
    /**
     * @brief Executes the pipeline using native OpenCV baseline modules.
     */
    static bool runPipelineOpenCV(
        const cv::Mat &gray1,
        const cv::Mat &gray2,
        const cv::Mat &bgr1,
        const cv::Size &sz,
        const cv::Mat &K,
        PipelineResult &res);

    /**
     * @brief Executes the pipeline using your custom hand-rolled mathematical backends.
     * @note Placeholder ready for when you substitute parts (like custom RANSAC, 8-point, or custom rectification).
     */
    static bool runPipelineCustom(
        const cv::Mat &gray1,
        const cv::Mat &gray2,
        const cv::Mat &bgr1,
        const cv::Size &sz,
        const cv::Mat &K,
        PipelineResult &res);
};