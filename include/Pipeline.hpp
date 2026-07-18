#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <eigen3/Eigen/Dense>
#include "DTULoader.hpp"
#include "Triangulation.hpp"
#include "Disparity.hpp"
#include "PipelineConfig.hpp"

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
    cv::Mat disparityConfidence; // CV_32F c_stereo in [0,1]
    cv::Mat dense3DPoints;  // CV_32FC3 spatial point grid for downstream ICP pipelines
    float globalConfidence = 1.0f; // ratio of RANSAC inliers to total sparse matches
};

/**
 * @class Pipeline
 * @brief Orchestration engine coordinating feature matching, epipolar geometry,
 * stereo rectification, and dense matching search-bound configuration.
 *
 * Every step with alternative backends (fundamental matrix, rectification,
 * disparity, triangulation) is selected via PipelineConfig; sparse matching
 * and pose recovery have a single implementation.
 */
class Pipeline
{
public:
    /**
     * @brief Runs the full stereo reconstruction pipeline with per-step backends taken from config.
     * @return true if the entire orchestration sequence finishes successfully, false otherwise.
     */
    static bool runPipeline(
        const cv::Mat &imgLeft,
        const cv::Mat &imgRight,
        const cv::Mat &K,
        PipelineResult &res,
        const PipelineConfig &config = PipelineConfig(),
        const Eigen::Vector3d &C1 = Eigen::Vector3d::Zero(),
        const Eigen::Vector3d &C2 = Eigen::Vector3d::Zero());

    /**
     * @brief Grayscale conversion + config.processingScale downscale.
     */
    static void preprocessScale(
        const cv::Mat &imgLeft, const cv::Mat &imgRight, const cv::Mat &K_in, double scale,
        cv::Mat &gray1, cv::Mat &gray2, cv::Mat &bgrLeft, cv::Mat &bgrRight, cv::Mat &K, cv::Size &sz);

    /**
     * @brief Scales a disparity block size (in pixels) by `scale` so the matching window
     * covers roughly the same extent of the scene regardless of config.processingScale.
     */
    static int scaledBlockSize(int baseBlockSize, double scale);

    /**
     * @brief Computes true metric baseline ||C1 - C2|| and rescales t in-place (t = t_unit * trueBaseline)
     */
    static void rescaleToTrueBaseline(const Eigen::Vector3d &C1, const Eigen::Vector3d &C2, cv::Mat &t);
};
