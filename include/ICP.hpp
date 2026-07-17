#pragma once

#include "PlyUtils.hpp"
#include "NearestNeighborSearch.hpp"
#include <Eigen/Dense>
#include <vector>

/**
 * @enum ICPMode
 * @brief Selects the residual formulation used by the ICP optimizer.
 */
enum class ICPMode {
    PointToPoint,
    PointToPlane
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

protected:
    bool     m_usePointToPlane;
    bool     m_useWeights;
    bool     m_verbose;
    unsigned m_nIterations;
    int      m_lastMatchCount;

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
