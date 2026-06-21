#include "Evaluator.hpp"
#include <cmath>
#include <algorithm>

void Evaluator::evaluatePose(const Eigen::Matrix3d &R_est, const Eigen::Vector3d &t_est,
                             const Eigen::Matrix3d &R_gt, const Eigen::Vector3d &t_gt,
                             double &rot_error_deg, double &trans_error_deg)
{
    const double PI = std::acos(-1.0);

    // calculate difference using Rotation matrix property.
    Eigen::Matrix3d R_diff = R_est * R_gt.transpose();
    double trace = R_diff.trace();

    // angle = arccos(Trace(R_diff) - 1 / 2)
    double cos_theta = (trace - 1.0) / 2.0;

    // clamp to avoid floating point inaccuracies
    cos_theta = std::clamp(cos_theta, -1.0, 1.0);
    rot_error_deg = std::acos(cos_theta) * (180.0 / PI);

    // translation angular error
    Eigen::Vector3d t_est_norm = t_est.normalized();
    Eigen::Vector3d t_gt_norm = t_gt.normalized();
    double dot_prod = t_est_norm.dot(t_gt_norm);

    dot_prod = std::clamp(dot_prod, -1.0, 1.0);
    trans_error_deg = std::acos(dot_prod) * (180.0 / PI);
}