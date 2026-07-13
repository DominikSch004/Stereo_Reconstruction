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
 * @struct PointConfidenceBreakdown
 * @brief Optional per-point decomposition of the confidence weight, emitted by
 *        buildPointCloud for the confidence-ICP ablation (benchmark/ConfidenceAnalysis.cpp).
 *
 * PointCloud::weights only stores the PRODUCT
 *   w = globalConfidence * c_depth * c_edge * c_stereo,
 * which makes it impossible to ask which factor carries the signal. When a
 * breakdown pointer is supplied, each factor is pushed in lockstep with
 * PointCloud::pts (same order, same filtering), so the arrays stay index-aligned.
 * The per-cloud globalConfidence factor is a scalar and is NOT stored here.
 */
struct PointConfidenceBreakdown
{
    std::vector<float> camDepth;   // Z in the rectified-left camera frame (mm); drives c_depth via sigma_Z ~ Z^2
    std::vector<float> depthConf;  // c_depth: median-normalized inverse depth variance, capped 100:1
    std::vector<float> edgeConf;   // c_edge: exp(-|grad disparity| / 5)
    std::vector<float> stereoConf; // c_stereo: left/right + photometric reliability in [0,1] (1.0 if no map supplied)

    void clear() { camDepth.clear(); depthConf.clear(); edgeConf.clear(); stereoConf.clear(); }
    void reserve(size_t n) { camDepth.reserve(n); depthConf.reserve(n); edgeConf.reserve(n); stereoConf.reserve(n); }
};

/**
 * @struct ConfidenceWeightConfig
 * @brief Per-factor on/off switches for composing the confidence weight in
 *        buildPointCloud. A disabled factor contributes a neutral 1.0, so
 *        {depth only} yields w = c_depth and {all off} yields uniform w = 1.
 *
 * This is the primitive behind the confidence->ICP factor ablation
 * ([[Confidence-ICP-Ablation]]): the diagnostic core found the PRODUCT of all
 * factors is uninformative because the informative c_depth is diluted by the
 * (mildly inverted) c_edge / c_stereo, so being able to compose w from a subset
 * of factors is what lets the ablation isolate the useful signal. The raw
 * per-factor values are still recorded in PointConfidenceBreakdown regardless of
 * which factors compose w. Default = all on = the historical behavior.
 */
struct ConfidenceWeightConfig
{
    bool useGlobal = true; // c_global (per-cloud F-inlier ratio)
    bool useDepth  = true; // c_depth  (inverse depth variance)
    bool useEdge   = true; // c_edge   (disparity-gradient suppression)
    bool useStereo = true; // c_stereo (left/right + photometric)
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
        TriangulationMethod method = TriangulationMethod::OpenCV,
        const cv::Mat &disparityConfidence = cv::Mat()
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
        TriangulationMethod method = TriangulationMethod::OpenCV,
        const cv::Mat &disparityConfidence = cv::Mat(),
        const ConfidenceWeightConfig &weightCfg = ConfidenceWeightConfig(),
        PointConfidenceBreakdown *breakdown = nullptr
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
     * @brief Confidence-weighted voxel downsampling. Each occupied voxel produces
     * one averaged point with consistently averaged color/normal attributes.
     */
    static PointCloud voxelDownsample(const PointCloud &cloud, float voxelSize);

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
};
