#pragma once

#include "PlyUtils.hpp"
#include <Eigen/Dense>

/**
 * @enum ICPMode
 * @brief Selects the residual formulation used by ICP::align.
 */
enum class ICPMode {
    PointToPoint,
    PointToPlane
};

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
     * @param mode Point-to-point or point-to-plane residual formulation.
     * @return 4x4 Homogeneous incremental rigid registration transformation matrix.
     */
    static Eigen::Matrix4f align(
        PointCloud& source,
        const PointCloud& target,
        int maxIter = 30,
        float distThresh = 0.1f,
        bool useWeights = true,
        ICPMode mode = ICPMode::PointToPoint
    );
};