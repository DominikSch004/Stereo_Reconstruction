#pragma once

#include <string>
#include <utility>
#include <vector>
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
#include "Disparity.hpp"
#include "Triangulation.hpp"
#include "ICP.hpp"
#include "PlyUtils.hpp"

enum class FeatureDetector
{
    SIFT,
    ORB
};

/**
 * @struct PipelineConfig
 * @brief Per-step backend selection for the stereo reconstruction pipeline.
 *
 * Each configurable step maps to one of the existing component enums. Steps
 * Essential matrix / pose recovery has no alternative implementation and is
 * intentionally not configurable.
 */
struct PipelineConfig
{

    FundamentalMethod fundamental = FundamentalMethod::OpenCVRANSAC;
    FeatureDetector featureDetector = FeatureDetector::SIFT;
    RectificationMethod rectification = RectificationMethod::CalibratedOpenCV;
    DisparityMethod disparity = DisparityMethod::OpenCVSGBM;
    TriangulationMethod triangulation = TriangulationMethod::OpenCV;
    ICPMode icpMode = ICPMode::PointToPoint; // used by the ICP fusion pipeline only
    bool stereoConfidenceFilter = true;
    float stereoLRMaxDiff = 1.5f;
    float stereoPhotometricScale = 25.0f;
    bool confidenceUseGlobal = true;
    bool confidenceUseDepth = true;
    bool confidenceUseEdge = true;
    bool confidenceUseStereo = true;
    int rngSeed = 42;
    mutable std::mt19937 rng;

    // Non-linear refinement of (R, t) directly on the essential-matrix space
    // (see GeometryUtils::refinePose). See FundamentalMatrix.cpp comments for details.
    // This flag is just for initial comparision and should later be removed
    bool refinePose = false;

    // dataset and image selection
    int datasetId = 1;
    int imageLeftId = 1;
    int imageRightId = 2;
    int illuminationId = 3;

    // Image pairs fused by the IcpFusion executable. Filled from
    // 'icp_image_pairs' (explicit list) or 'icp_view_range' (expanded to
    // consecutive pairs). Default: consecutive pairs over views 6..19.
    std::vector<std::pair<int, int>> icpImagePairs;

    // threshold for FLANN after SIFT
    float ratioThreshold = 0.75f;

    // Image downscale factor applied before every pipeline step, via
    // Pipeline::preprocessScale. 1.0 = no downscale. Lower values trade
    // accuracy/coverage for speed. Working at 0.5 for speed.
    float processingScale = 0.5f;

    // Custom disparity backend only: toggles/thresholds for the optional
    // post-processing stages on top of the core disparity algorithm
    // (Hirschmuller 2008 Sec 2.5.1, 2.5.3).
    DisparityRefinementConfig disparityRefinement;

    /**
     * @brief Loads a config from a YAML file (parsed with cv::FileStorage).
     * @throws std::runtime_error if the file cannot be opened or a key holds an unknown value.
     * Missing keys keep their defaults (a note is printed).
     */
    static PipelineConfig load(const std::string &path);

    /** @brief Prints the selected backend of every step. */
    void print() const;

    ConfidenceWeightConfig confidenceWeights() const
    {
        return {confidenceUseGlobal, confidenceUseDepth,
                confidenceUseEdge, confidenceUseStereo};
    }
};
