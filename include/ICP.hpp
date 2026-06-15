#pragma once

#include "Cloud.hpp"
#include <Eigen/Dense>

class ICP
{
public:

    static Eigen::Matrix4f align(
        Cloud& source,
        const Cloud& target,
        int maxIter = 30,
        float distThresh = 0.1f);
};