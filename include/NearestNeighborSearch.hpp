#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/flann.hpp>

/**
 * @struct Match
 * @brief A single source->target correspondence produced by a nearest-neighbor query.
 */
struct Match
{
    int   idx;       ///< Index into the target point set, or -1 if no valid match.
    float distance;  ///< Squared Euclidean distance to the matched target point.
};

/**
 * @class NearestNeighborSearch
 * @brief FLANN KD-tree wrapper for source->target correspondence lookup.
 *
 * The target point set is indexed once via buildIndex(); every subsequent
 * queryMatches() call reuses that index. Matches farther than the configured
 * maximum distance are returned with idx == -1 (invalid), which keeps the
 * distance-gating policy in one place instead of scattered across callers.
 */
class NearestNeighborSearch
{
public:
    /**
     * @brief Sets the maximum (Euclidean) distance a correspondence may span.
     *        Queries beyond this radius are reported as invalid (idx == -1).
     */
    void setMatchingMaxDistance(float maxDistance) { m_maxDistanceSq = maxDistance * maxDistance; }

    /**
     * @brief Builds the KD-tree over the target points. Must be called before queryMatches().
     */
    void buildIndex(const std::vector<Eigen::Vector3f>& targetPoints);

    /**
     * @brief Returns the nearest target point for each query point, gated by the max distance.
     */
    std::vector<Match> queryMatches(const std::vector<Eigen::Vector3f>& queryPoints) const;

private:
    cv::Mat                             m_targetMat;      ///< Kept alive for the index to reference.
    std::unique_ptr<cv::flann::Index>   m_index;
    float                               m_maxDistanceSq = std::numeric_limits<float>::max();
};
