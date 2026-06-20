#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

/**
 * @class FundamentalMatrix
 * @brief Handles epipolar geometry estimation between two unrectified views.
 * * Provides utilities to compute the 3x3 Fundamental Matrix (F) using either
 * a custom normalized 8-point RANSAC pipeline or OpenCV's native robust estimators.
 */
class FundamentalMatrix
{
public:
    /**
     * @brief Computes the Sampson distance error for a single point pair.
     * @details Acts as a first-order geometric approximation of the distance to the 
     * epipolar line. Used as the error metric for RANSAC inlier detection.
     * @param F 3x3 Fundamental matrix mapping points from Left to Right image.
     * @param pl 2D keypoint in the left image.
     * @param pr 2D keypoint in the right image.
     * @return Squared Sampson distance error (double).
     */
    static double sampsonError(
        const Eigen::Matrix3d& F,
        const cv::Point2f& pl,
        const cv::Point2f& pr);

    /**
     * @brief Applies Hartley normalization to a set of 2D image coordinates.
     * @details Translates points to their centroid (zero-mean) and scales them so 
     * the average distance to the origin is sqrt(2). This conditions the 
     * design matrix A to prevent numerical instability during SVD.
     * @param pts Input vector of unnormalized 2D image coordinates.
     * @param[out] ptsNorm Output vector populated with the normalized 2D points.
     * @return 3x3 Homogeneous transformation matrix (T) used for normalization.
     */
    static Eigen::Matrix3d normalizePoints(
        const std::vector<cv::Point2f>& pts,
        std::vector<cv::Point2f>& ptsNorm);

    /**
     * @brief Core algebraic linear solver for the 8-point algorithm.
     * @details Solves the homogeneous system Af = 0 using Singular Value Decomposition (SVD),
     * enforces the rank-2 constraint on F by zeroing out the third singular value,
     * and denormalizes the result back to pixel coordinates.
     * @note Requires at least 8 distinct point correspondences.
     * @param ptsL Core matched points from the left image.
     * @param ptsR Core matched points from the right image.
     * @return Rank-2 constrained 3x3 Fundamental Matrix.
     */
    static Eigen::Matrix3d compute8Point(
        const std::vector<cv::Point2f>& ptsL,
        const std::vector<cv::Point2f>& ptsR);

    /**
     * @brief APPROACH A (Custom): Estimates F robustly using a manual RANSAC loop.
     * @details Randomly samples 8-point subsets to generate hypotheses, evaluates
     * inliers via the Sampson distance threshold, and performs a final 
     * Levenberg-Marquardt style refit over the entire gathered inlier set.
     * @param ptsL All matched features in the left image.
     * @param ptsR All matched features in the right image.
     * @param[out] inlierMask Array tracking true/false status for each input pair.
     * @param threshold Max allowed Sampson distance error to count as an inlier (pixels).
     * @param maxIter Maximum allocation of sampling loop generations.
     * @return Refined 3x3 Fundamental Matrix.
     */
    static Eigen::Matrix3d computeCustomRANSAC(
        const std::vector<cv::Point2f>& ptsL,
        const std::vector<cv::Point2f>& ptsR,
        std::vector<bool>& inlierMask,
        double threshold = 1.0,
        int maxIter = 1000);

    /**
     * @brief APPROACH B (OpenCV): Wrapper for cv::findFundamentalMat.
     * @details Utilizes OpenCV's highly optimized, multi-threaded RANSAC execution 
     * path to act as a gold-standard baseline for custom algorithm evaluation.
     * @param ptsL All matched features in the left image.
     * @param ptsR All matched features in the right image.
     * @param[out] inlierMask Vector tracking true/false status for each input pair.
     * @param threshold Maximum allowed epipolar distance error (pixels).
     * @param confidence Target probability configuration (0.0 to 1.0) for completion hooks.
     * @return Converted Eigen::Matrix3d representation of OpenCV's computed F.
     */
    static Eigen::Matrix3d computeOpenCVRANSAC(
        const std::vector<cv::Point2f>& ptsL,
        const std::vector<cv::Point2f>& ptsR,
        std::vector<bool>& inlierMask,
        double threshold = 1.0,
        double confidence = 0.99);

    /**
     * NOTE ON THE ESSENTIAL MATRIX (E):
     * To extract the actual relative camera rotation (R) and translation (t) 
     * for downstream ICP/reconstruction seeding, compute the Essential Matrix:
     * * Eigen::Matrix3d E = K_right.transpose() * F * K_left;
     * * Where K_left and K_right are the camera intrinsic calibration matrices.
     * Ensure this step is completed before initiating triangulation.
     */
};