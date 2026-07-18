#pragma once

#include <vector>
#include <random>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "Triangulation.hpp"

/**
 * @struct PointCloud
 * @brief Container holding metric 3D point vectors and corresponding RGB texture attributes.
 */
struct PointCloud
{
    std::vector<Eigen::Vector3f> pts;
    std::vector<cv::Vec3b>       colors;
    std::vector<float> weights;
    std::vector<Eigen::Vector3f> normals; // per-point surface normal
    std::vector<bool> validNormal;        // true if normals[i] was computed from a well-conditioned local neighborhood
};

/**
 * @class PlyUtils
 * @brief Coordinates dense point cloud generation, coordinate transformations, and PLY streaming.
 */
class PlyUtils
{
public:
    /**
     * @brief High-level orchestration utility that builds a point cloud from stereo results
     * and streams it immediately to disk.
     * @return true if file writing succeeds, false otherwise.
     */
    static bool buildAndSavePLY(
        const std::string &path,
        const cv::Mat &disparity,
        const cv::Mat &Q,
        const cv::Mat &P1r,
        const cv::Mat &P2r,
        const cv::Mat &camToWorld,
        const cv::Mat &rectColor,
        int minDisp,
        float globalConfidence,
        TriangulationMethod method = TriangulationMethod::OpenCV
    );

    /**
     * @brief Generates a dense 3D point cloud from a disparity map and filters out noisy artifacts.
     * @param disparity Input disparity image map.
     * @param Q 4x4 reprojection matrix.
     * @param P1r 3x4 rectified left projection matrix.
     * @param P2r 3x4 rectified right projection matrix.
     * @param camToWorld 3x4 or 4x4 rigid transformation tracking extrinsic placement.
     * @param rectColor The rectified left image used to sample color data.
     * @param minDisp The threshold used to skip uncalculated/background disparities.
     * @param globalConfidence Pair-level inlier ratio input to the per-point weighting model --
     *  see computePointWeight.
     * @param method Strategy selected for triangulation calculation.
     */
    static PointCloud buildPointCloud(
        const cv::Mat &disparity,
        const cv::Mat &Q,
        const cv::Mat &P1r,
        const cv::Mat &P2r,
        const cv::Mat &camToWorld,
        const cv::Mat &rectColor,
        int minDisp,
        float globalConfidence,
        TriangulationMethod method = TriangulationMethod::OpenCV
    );

    /**
     * @brief Writes point cloud data to an ASCII PLY format target.
     */
    static void savePLY(
        const std::string &path,
        const PointCloud &cloud);

    static PointCloud loadPLY(const std::string &path);

    /**
     * @brief Subsamples point indices randomly to support real-time ICP requirements.
     */
    static PointCloud subsample(
        const PointCloud &cloud,
        size_t n,
        std::mt19937 &rng);

    /**
     * @brief Normalizes point spatial attributes to zero-mean and unit variance.
     */
    static std::pair<Eigen::Vector3f, float> normalise(PointCloud &cloud);

    /**
     * @brief Reverts normalization scaling.
     */
    static void denormalise(
        PointCloud& cloud,
        const Eigen::Vector3f& mean,
        float scale
    );

private:
    /**
     * @brief Combines three independent confidence signals into one per-point weight:
     *  - depth uncertainty propagated from disparity noise: variance = Z^4/(f*B)^2 * sigmaD^2,
     *    confidence = 1/(1+variance) -- farther/noisier points trusted less.
     *  - a discontinuity penalty from the local disparity-gradient magnitude (exponential
     *    decay, ~1.0 on smooth surfaces, drops toward 0 at depth edges/occlusion boundaries).
     *  - the pair's global robust-estimator inlier ratio (RANSAC/MAGSAC/PROSAC, whichever
     *    config.fundamental selected; constant across the cloud).
     * @param z Triangulated depth (mm) of the point.
     * @param fB Focal length * baseline (from P2r(0,3) = -f*B), used by the depth-uncertainty term.
     * @param edgeGradient Disparity-map gradient magnitude at this pixel (Sobel).
     * @param globalConfidence Pair-level robust-estimator inlier ratio.
     */
    static float computePointWeight(
        float z, double fB, float edgeGradient,
        float globalConfidence
    );
};
