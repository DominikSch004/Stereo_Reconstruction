#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <random>

/**
 * @enum FundamentalMethod
 * @brief Selects the robust estimation engine to compute the Fundamental Matrix.
 */
enum class FundamentalMethod
{
    CustomRANSAC, // Custom hand-rolled 8-point RANSAC loop with Sampson distance refitting
    OpenCVRANSAC, // Native, multi-threaded OpenCV robust solver baseline
    CustomMAGSAC, // Custom implementation of the MAGSAC robust estimator
    CustomPROSAC  // Custom implementation of the PROSAC robust estimator
};

struct VisualizationData
{
    std::vector<std::vector<cv::Point2f>> history_L;
    std::vector<std::vector<cv::Point2f>> history_R;
    std::vector<int> inLierCount;
    std::vector<int> currBestInlier;
    double final_sampson_err;
    int final_inlier_count;
};

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
     * @brief High-level unified entrypoint to estimate F robustly using a selectable backend.
     * @param ptsL All matched features in the left image.
     * @param ptsR All matched features in the right image.
     * @param[out] inlierMask Array tracking true/false status for each input pair.
     * @param method Choice of robust estimation engine (CustomRANSAC or OpenCVRANSAC).
     * @param threshold Max allowed epipolar/Sampson distance error to count as an inlier (pixels).
     * @param confidence Target probability configuration (0.0 to 1.0) for RANSAC completion hooks (OpenCV only).
     * @param maxIter Maximum allocation of sampling loop generations (CustomRANSAC only).
     * @return Refined 3x3 Fundamental Matrix (Eigen::Matrix3d).
     */
    static Eigen::Matrix3d computeFundamental(
        const std::vector<cv::Point2f> &ptsL,
        const std::vector<cv::Point2f> &ptsR,
        std::vector<bool> &inlierMask,
        std::mt19937 &rng,
        VisualizationData &visualize,
        FundamentalMethod method = FundamentalMethod::OpenCVRANSAC,
        double threshold = 1.0,
        double confidence = 0.99,
        int maxIter = 1000);

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
        const Eigen::Matrix3d &F,
        const cv::Point2f &pl,
        const cv::Point2f &pr);

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
        const std::vector<cv::Point2f> &pts,
        std::vector<cv::Point2f> &ptsNorm);

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
        const std::vector<cv::Point2f> &ptsL,
        const std::vector<cv::Point2f> &ptsR);

private:
    /**
     * @brief Custom APPROACH : Estimates F robustly using a manual RANSAC loop.
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
        const std::vector<cv::Point2f> &ptsL,
        const std::vector<cv::Point2f> &ptsR,
        std::vector<bool> &inlierMask,
        std::mt19937 &rng,
        VisualizationData &visualize,
        double threshold = 1.0,
        double confidence = 0.99,
        int maxIter = 1000);

    /**
     * @brief OpenCV APPROACH : Wrapper for cv::findFundamentalMat.
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
        const std::vector<cv::Point2f> &ptsL,
        const std::vector<cv::Point2f> &ptsR,
        std::vector<bool> &inlierMask,
        double threshold = 1.0,
        double confidence = 0.99);

    /**
     * @brief Weighted variant of the normalized 8-point solver.
     * @details Builds the same linear epipolar system as compute8Point(), but
     * scales each correspondence row by sqrt(weight). Used by MAGSAC after
     * soft inlier scoring so stronger correspondences influence the final SVD
     * estimate more than weak ones.
     * @param ptsL Matched points from the left image.
     * @param ptsR Matched points from the right image.
     * @param weights Per-correspondence reliability weights.
     * @return Rank-2 constrained 3x3 Fundamental Matrix.
     */
    static Eigen::Matrix3d computeWeighted8Point(
        const std::vector<cv::Point2f> &ptsL,
        const std::vector<cv::Point2f> &ptsR,
        const std::vector<double> &weights);

    /**
     * @brief Custom APPROACH : Estimates F using a MAGSAC-style robust loop.
     * @details Samples 8-point hypotheses, scores each candidate with a soft
     * truncated Gaussian function of the Sampson error, and refits the final
     * model with weighted 8-point estimation over the retained correspondences.
     * @param ptsL All matched features in the left image.
     * @param ptsR All matched features in the right image.
     * @param[out] inlierMask Array tracking true/false status for each input pair.
     * @param sigmaMax Maximum assumed noise scale for soft scoring.
     * @param maxIter Maximum allocation of sampling loop generations.
     * @return Refined 3x3 Fundamental Matrix.
     */
    static Eigen::Matrix3d computeCustomMAGSAC(
        const std::vector<cv::Point2f> &ptsL,
        const std::vector<cv::Point2f> &ptsR,
        std::vector<bool> &inlierMask,
        std::mt19937 &rng,
        double sigmaMax,
        double confidence,
        int maxIter);

    /**
     * @brief Custom APPROACH : Estimates F using a PROSAC-style sampling loop.
     * @details Assumes matches are ordered by quality, samples first from the
     * highest-ranked subset, progressively expands the sampling pool, evaluates
     * every candidate on all matches, and refits using all geometric inliers.
     * @param ptsL All matched features in the left image.
     * @param ptsR All matched features in the right image.
     * @param[out] inlierMask Array tracking true/false status for each input pair.
     * @param threshold Max allowed Sampson distance error to count as an inlier.
     * @param maxIter Maximum allocation of sampling loop generations.
     * @return Refined 3x3 Fundamental Matrix.
     */
    static Eigen::Matrix3d computeCustomPROSAC(
        const std::vector<cv::Point2f> &ptsL,
        const std::vector<cv::Point2f> &ptsR,
        std::vector<bool> &inlierMask,
        std::mt19937 &rng,
        VisualizationData &visualize,
        double threshold,
        double confidence,
        int maxIter);

    /**
     * @brief Helper function to calculate dynamic U-SAC iterations.
     */
    static int calculateRequiredIterations(
        int bestInliers,
        int N,
        int sampleSize,
        double confidence);

    /**
     * NOTE ON THE ESSENTIAL MATRIX (E):
     * To extract the actual relative camera rotation (R) and translation (t)
     * for downstream ICP/reconstruction seeding, compute the Essential Matrix:
     * * Eigen::Matrix3d E = K_right.transpose() * F * K_left;
     * * Where K_left and K_right are the camera intrinsic calibration matrices.
     * Ensure this step is completed before initiating triangulation.
     */

    // Add the planar degeneracy filter
    static bool isCoplanar(const std::vector<cv::Point2f> &sL,
                           const std::vector<cv::Point2f> &sR, const double ransacThreshold = 1.5);
};
