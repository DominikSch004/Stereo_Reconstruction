#include "VoxelFusion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

VoxelFusionModel::VoxelFusionModel(float voxelSize, float normalOutlierFactor)
    : m_voxelSize(std::max(voxelSize, 1e-6f)),
      m_invVoxelSize(1.0f / m_voxelSize),
      m_normalOutlierDistance(std::max(normalOutlierFactor, 0.1f) * m_voxelSize)
{
}

size_t VoxelFusionModel::KeyHash::operator()(const Key& k) const
{
    size_t h = std::hash<int>{}(k.x);
    h ^= std::hash<int>{}(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

VoxelFusionModel::Key VoxelFusionModel::keyFor(const Eigen::Vector3f& p) const
{
    return {static_cast<int>(std::floor(p.x() * m_invVoxelSize)),
            static_cast<int>(std::floor(p.y() * m_invVoxelSize)),
            static_cast<int>(std::floor(p.z() * m_invVoxelSize))};
}

void VoxelFusionModel::addToSurfel(Surfel& s, const PointCloud& cloud, size_t i, double w)
{
    s.positionSum += w * cloud.pts[i].cast<double>();
    if (i < cloud.colors.size())
        s.colorSum += w * Eigen::Vector3d(cloud.colors[i][0], cloud.colors[i][1], cloud.colors[i][2]);

    if (i < cloud.normals.size() &&
        (i >= cloud.validNormal.size() || cloud.validNormal[i]) &&
        cloud.normals[i].allFinite() && cloud.normals[i].squaredNorm() > 1e-12f)
    {
        Eigen::Vector3d n = cloud.normals[i].cast<double>().normalized();
        if (s.hasNormal && s.normalSum.dot(n) < 0.0) n = -n;
        s.normalSum += w * n;
        s.hasNormal = true;
    }
    s.precision += w;
    ++s.observations;
}

VoxelFusionModel::UpdateStats VoxelFusionModel::integrate(const PointCloud& cloud)
{
    UpdateStats stats;
    m_voxels.reserve(m_voxels.size() + cloud.pts.size() / 2);
    const double neighborRadiusSq = 2.25 * m_voxelSize * m_voxelSize;

    for (size_t i = 0; i < cloud.pts.size(); ++i)
    {
        if (!cloud.pts[i].allFinite()) { ++stats.rejected; continue; }
        const double w = (i < cloud.weights.size() && std::isfinite(cloud.weights[i]))
                       ? std::max<double>(cloud.weights[i], 1e-6) : 1.0;
        const Key own = keyFor(cloud.pts[i]);

        auto best = m_voxels.end();
        double bestDistSq = std::numeric_limits<double>::infinity();
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    auto it = m_voxels.find({own.x + dx, own.y + dy, own.z + dz});
                    if (it == m_voxels.end()) continue;
                    const double d2 = (cloud.pts[i].cast<double>() - it->second.position()).squaredNorm();
                    if (d2 < bestDistSq) { bestDistSq = d2; best = it; }
                }

        if (best != m_voxels.end() && bestDistSq <= neighborRadiusSq)
        {
            const Surfel& s = best->second;
            if (s.hasNormal)
            {
                const double normalResidual = std::abs(
                    s.normal().dot(cloud.pts[i].cast<double>() - s.position()));
                if (normalResidual > m_normalOutlierDistance)
                {
                    ++stats.rejected;
                    continue;
                }
            }
            addToSurfel(best->second, cloud, i, w);
            ++stats.merged;
        }
        else
        {
            Surfel& s = m_voxels[own];
            addToSurfel(s, cloud, i, w);
            ++stats.inserted;
        }
    }
    return stats;
}

PointCloud VoxelFusionModel::pointCloud() const
{
    PointCloud out;
    out.pts.reserve(m_voxels.size()); out.colors.reserve(m_voxels.size());
    out.weights.reserve(m_voxels.size()); out.normals.reserve(m_voxels.size());
    out.validNormal.reserve(m_voxels.size());
    for (const auto& [key, s] : m_voxels)
    {
        (void)key;
        out.pts.push_back(s.position().cast<float>());
        const Eigen::Vector3d c = s.colorSum / std::max(s.precision, 1e-12);
        out.colors.emplace_back(cv::saturate_cast<uchar>(c.x()),
                                cv::saturate_cast<uchar>(c.y()),
                                cv::saturate_cast<uchar>(c.z()));
        out.weights.push_back(static_cast<float>(s.precision));
        if (s.hasNormal && s.normalSum.squaredNorm() > 1e-20)
        {
            out.normals.push_back(s.normal().cast<float>());
            out.validNormal.push_back(true);
        }
        else
        {
            out.normals.push_back(Eigen::Vector3f::Zero());
            out.validNormal.push_back(false);
        }
    }
    return out;
}
