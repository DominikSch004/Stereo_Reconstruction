#pragma once

#include "PlyUtils.hpp"
#include <Eigen/Dense>

/**
 * @class ICP
 * @brief ICP: Iterative Closest Point for rigid point cloud registration.
 */
class ICP
{
public:
    /**
     * @brief Aligns a mutable source point cloud into a target point cloud frame.
     * @param source The moving point cloud (modified in place by the calculated transforms).
     * @param target The reference fixed point cloud.
     * @return 4x4 Homogeneous incremental rigid registration transformation matrix.
     */
    static Eigen::Matrix4f align(
        PointCloud& source,
        const PointCloud& target,
        int maxIter = 30,
        float distThresh = 0.1f,
        bool useWeights = true
    );
};