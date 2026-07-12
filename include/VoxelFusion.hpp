#pragma once

#include "PlyUtils.hpp"
#include <algorithm>
#include <cstddef>
#include <unordered_map>

/**
 * @brief Incremental confidence-weighted surfel/voxel fusion model.
 *
 * Repeated observations in the same local surface region are averaged instead of
 * appended. Nearby observations that disagree too far along the accumulated normal
 * are rejected, preventing isolated stereo spikes from thickening the surface.
 */
class VoxelFusionModel
{
public:
    struct UpdateStats { size_t inserted = 0, merged = 0, rejected = 0; };

    VoxelFusionModel(float voxelSize, float normalOutlierFactor = 1.5f);
    UpdateStats integrate(const PointCloud& cloud);
    PointCloud pointCloud() const;
    size_t size() const { return m_voxels.size(); }

private:
    struct Key
    {
        int x, y, z;
        bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct KeyHash
    {
        size_t operator()(const Key& k) const;
    };
    struct Surfel
    {
        Eigen::Vector3d positionSum = Eigen::Vector3d::Zero();
        Eigen::Vector3d normalSum = Eigen::Vector3d::Zero();
        Eigen::Vector3d colorSum = Eigen::Vector3d::Zero();
        double precision = 0.0;
        size_t observations = 0;
        bool hasNormal = false;

        Eigen::Vector3d position() const { return positionSum / std::max(precision, 1e-12); }
        Eigen::Vector3d normal() const {
            return hasNormal && normalSum.squaredNorm() > 1e-20
                 ? normalSum.normalized() : Eigen::Vector3d::Zero();
        }
    };

    Key keyFor(const Eigen::Vector3f& p) const;
    void addToSurfel(Surfel& surfel, const PointCloud& cloud, size_t i, double weight);

    float m_voxelSize;
    float m_invVoxelSize;
    float m_normalOutlierDistance;
    std::unordered_map<Key, Surfel, KeyHash> m_voxels;
};
