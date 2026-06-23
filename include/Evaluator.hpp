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

    // Computes the Symmetric Epipolar Distance (in pixels) for a set of point correspondences.
    // This measures the geometric 2D fit by calculating the perpendicular distance from
    // each point to its corresponding epipolar line in both images.
    static double computeSymmetricEpipolarDistance(const std::vector<cv::Point2f> &pts1,
                                                   const std::vector<cv::Point2f> &pts2,
                                                   const cv::Mat &F);

    static double evaluateEpipolarError(const Eigen::Matrix3d &F_eigen,
                                        const std::vector<cv::Point2f> &ptsL,
                                        const std::vector<cv::Point2f> &ptsR,
                                        const std::vector<bool> &inlierMask);

    // Calculates the percentage of matched points that survived the RANSAC filtering process.
    static double computeInlierRatio(const std::vector<bool> &inlierMask);
};