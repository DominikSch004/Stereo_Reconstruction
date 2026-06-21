#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

class Evaluator
{
public:
    // Computes the absolute geometric error between an estimated pose and ground truth.
    // Rotation Error: The geodesic distance (angle in degrees) required to align R_est with R_gt.
    // Translation Error: The scale-invariant angular difference between the directional vectors.
    static void evaluatePose(const Eigen::Matrix3d &R_est, const Eigen::Vector3d &t_est,
                             const Eigen::Matrix3d &R_gt, const Eigen::Vector3d &t_gt,
                             double &rot_error_deg, double &trans_error_deg);
};