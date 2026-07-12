#pragma once

#include <string>
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
#include "Disparity.hpp"
#include "Triangulation.hpp"
#include "ICP.hpp"

/**
 * @struct PipelineConfig
 * @brief Per-step backend selection for the stereo reconstruction pipeline.
 *
 * Each configurable step maps to one of the existing component enums. Steps
 * without an alternative implementation (sparse SIFT+FLANN matching, essential
 * matrix / pose recovery) are intentionally not configurable.
 */
struct PipelineConfig
{
    FundamentalMethod fundamental = FundamentalMethod::OpenCVRANSAC;
    RectificationMethod rectification = RectificationMethod::CalibratedOpenCV;
    DisparityMethod disparity = DisparityMethod::OpenCVSGBM;
    TriangulationMethod triangulation = TriangulationMethod::OpenCV;
    ICPMode icpMode = ICPMode::PointToPlane; // used by the ICP fusion pipeline only
    bool stereoConfidenceFilter = true;
    float stereoLRMaxDiff = 1.5f;
    float stereoPhotometricScale = 25.0f;
    bool icpRobust = true;
    bool icpReciprocal = true;
    float icpTrimFraction = 0.80f;
    float fusionVoxelSize = 0.01f;       // normalized cloud units
    float fusionOutlierFactor = 1.5f;    // normal-distance rejection in voxel units

    /**
     * @brief Loads a config from a YAML file (parsed with cv::FileStorage).
     * @throws std::runtime_error if the file cannot be opened or a key holds an unknown value.
     * Missing keys keep their defaults (a note is printed).
     */
    static PipelineConfig load(const std::string &path);

    /** @brief Prints the selected backend of every step. */
    void print() const;
};
