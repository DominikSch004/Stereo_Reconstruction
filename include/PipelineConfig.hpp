#pragma once

#include <string>
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
#include "Disparity.hpp"
#include "Triangulation.hpp"
#include "ICP.hpp"

/**
 * @enum IcpPairMode
 * @brief Which DTU view pairs the ICP fusion pipeline reconstructs and fuses.
 *   Selected    -> a small hand-picked list of horizontal pairs (fast, curated).
 *   Consecutive -> disjoint adjacent view pairs (i, i+1), (i+2, i+3), ... over
 *                  [icpViewFirst, icpViewLast], so each view feeds exactly one pair.
 * In both modes vertical baselines are still skipped by IcpUtils::orderPair.
 */
enum class IcpPairMode { Selected, Consecutive };

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
    // ICP fusion pair selection (used by the IcpFusion executable only)
    IcpPairMode icpPairMode = IcpPairMode::Selected;
    int icpViewFirst = 1;   // first DTU view id for consecutive-pair enumeration
    int icpViewLast = 49;   // last DTU view id (inclusive) for consecutive-pair enumeration
    bool stereoConfidenceFilter = true;
    float stereoLRMaxDiff = 1.5f;
    float stereoPhotometricScale = 25.0f;
    // Per-factor composition of the ICP confidence weight (see ConfidenceWeightConfig).
    // Default all-on = historical product. Set individual factors off to ablate them
    // (e.g. depth-only isolates the one informative factor found by the diagnostic core).
    bool confUseGlobal = true;
    bool confUseDepth = true;
    bool confUseEdge = true;
    bool confUseStereo = true;
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

    /** @brief Assembles the per-factor confidence-weight switches for buildPointCloud. */
    ConfidenceWeightConfig confidenceWeights() const
    {
        return {confUseGlobal, confUseDepth, confUseEdge, confUseStereo};
    }
};
