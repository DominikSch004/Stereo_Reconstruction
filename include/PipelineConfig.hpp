#pragma once

#include <string>
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
#include "Disparity.hpp"
#include "Triangulation.hpp"
#include "ICP.hpp"

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
    ICPMode icpMode = ICPMode::PointToPlane; // used by the ICP fusion pipeline only
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

    // threshold for FLANN after SIFT
    float ratioThreshold = 0.75f;
    /**
     * @brief Loads a config from a YAML file (parsed with cv::FileStorage).
     * @throws std::runtime_error if the file cannot be opened or a key holds an unknown value.
     * Missing keys keep their defaults (a note is printed).
     */
    static PipelineConfig load(const std::string &path);

    /** @brief Prints the selected backend of every step. */
    void print() const;
};
