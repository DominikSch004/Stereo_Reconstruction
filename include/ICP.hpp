#pragma once

#include "PlyUtils.hpp"
#include "NearestNeighborSearch.hpp"
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>

/**
 * @enum ICPMode
 * @brief Selects the residual formulation used by the ICP optimizer.
 */
enum class ICPMode {
    PointToPoint,
    PointToPlane
};

enum class ICPRobustLoss { None, Huber, Cauchy };

struct ICPMetrics
{
    int matchCount = 0;
    int sourceCount = 0;
    double overlap = 0.0;
    double meanDistance = std::numeric_limits<double>::infinity();
    double medianDistance = std::numeric_limits<double>::infinity();
    double rmse = std::numeric_limits<double>::infinity();
    double conditionNumber = std::numeric_limits<double>::infinity();
    double spatialCoverage = 0.0; // occupied source octants containing accepted matches
    bool valid() const { return matchCount >= 6 && std::isfinite(rmse); }
};

/**
 * @class ICPOptimizer
 * @brief Abstract base for Iterative Closest Point rigid registration.
 *
 * Holds the shared configuration (iteration count, correspondence distance,
 * residual mode, confidence weighting) and the geometry helpers common to any
 * ICP backend: point/normal transformation, correspondence pruning by normal
 * agreement, and the nearest-neighbor search. Concrete subclasses implement the
 * actual iterate-match-solve loop in estimatePose().
 */
class ICPOptimizer
{
public:
    ICPOptimizer();
    virtual ~ICPOptimizer() = default;

    /** @brief Maximum Euclidean distance a correspondence may span. */
    void setMatchingMaxDistance(float maxDistance);
    /** @brief Number of outer ICP iterations (re-association + solve). */
    void setNbOfIterations(unsigned nIterations);
    /** @brief Selects point-to-plane (true) or point-to-point (false) residuals. */
    void usePointToPlaneConstraints(bool enable);
    /** @brief Convenience wrapper accepting the ICPMode enum. */
    void setMode(ICPMode mode) { usePointToPlaneConstraints(mode == ICPMode::PointToPlane); }
    /** @brief Weight residuals by per-point source confidence when available. */
    void useWeights(bool enable);
    /** @brief Emit per-iteration correspondence / cost diagnostics. */
    void setVerbose(bool enable);
    void setRobustLoss(ICPRobustLoss loss, double scale = 0.02);
    void useReciprocalCorrespondences(bool enable);
    void setTrimFraction(float fraction);
    void useAdaptiveDistanceGate(bool enable);

    /**
     * @brief Estimates the rigid transform aligning @p source onto @p target.
     * @param source The moving point cloud. NOT modified -- the returned transform maps it onto target.
     * @param target The fixed reference point cloud.
     * @param initialPose Optional starting estimate to refine from.
     * @return 4x4 homogeneous rigid transform aligning source into the target frame.
     */
    virtual Eigen::Matrix4f estimatePose(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix4f& initialPose = Eigen::Matrix4f::Identity()) = 0;

    /**
     * @brief Correspondences retained on the last iteration (0 if ICP never ran).
     *        Lets callers gauge overlap and reject a pair that failed to register.
     */
    int lastMatchCount() const { return m_lastMatchCount; }
    const ICPMetrics& initialMetrics() const { return m_initialMetrics; }
    const ICPMetrics& finalMetrics() const { return m_finalMetrics; }

    /** @brief Evaluates a pose using the optimizer's current correspondence policy. */
    ICPMetrics evaluatePose(const PointCloud& source, const PointCloud& target,
                            const Eigen::Matrix4f& pose = Eigen::Matrix4f::Identity());

protected:
    bool     m_usePointToPlane;
    bool     m_useWeights;
    bool     m_verbose;
    unsigned m_nIterations;
    int      m_lastMatchCount;
    float    m_maxDistance;
    ICPRobustLoss m_robustLoss;
    double   m_robustScale;
    bool     m_useReciprocal;
    bool     m_useAdaptiveGate;
    float    m_trimFraction;
    ICPMetrics m_initialMetrics;
    ICPMetrics m_finalMetrics;

    NearestNeighborSearch m_nearestNeighborSearch;

    /** @brief Applies a rigid pose to every point. */
    std::vector<Eigen::Vector3f> transformPoints(
        const std::vector<Eigen::Vector3f>& points, const Eigen::Matrix4f& pose) const;

    /** @brief Rotates every normal by the pose's rotation block. */
    std::vector<Eigen::Vector3f> transformNormals(
        const std::vector<Eigen::Vector3f>& normals, const Eigen::Matrix4f& pose) const;

    /**
     * @brief Invalidates matches whose source/target normals disagree by more than 60 degrees.
     *        No-op when either cloud lacks per-point normals.
     */
    void pruneCorrespondences(
        const std::vector<Eigen::Vector3f>& sourceNormals,
        const std::vector<Eigen::Vector3f>& targetNormals,
        const std::vector<bool>& targetValidNormal,
        std::vector<Match>& matches) const;

    void robustlyFilterCorrespondences(
        const std::vector<Eigen::Vector3f>& transformedPoints,
        const PointCloud& target,
        std::vector<Match>& matches) const;

    ICPMetrics computeMetrics(
        const PointCloud& source,
        const PointCloud& target,
        const std::vector<Eigen::Vector3f>& transformedPoints,
        const std::vector<Match>& matches) const;
};

/**
 * @class CeresICPOptimizer
 * @brief ICP backend that solves each iteration's pose update with Ceres.
 */
class CeresICPOptimizer : public ICPOptimizer
{
public:
    Eigen::Matrix4f estimatePose(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix4f& initialPose = Eigen::Matrix4f::Identity()) override;
};
