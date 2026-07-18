#pragma once

#include <string>
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
