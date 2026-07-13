#pragma once

#include "PlyUtils.hpp"
#include "DTULoader.hpp"
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <random>

/**
 * @file IcpUtils.hpp
 * @brief Free-function helpers shared by the ICP fusion pipeline (src/IcpFusion.cpp)
 *        and its benchmarks (benchmark/IcpBenchmark.cpp, benchmark/IcpVerification.cpp):
 *        rigid-transform application, coordinate-frame placement, DTU view-pair
 *        ordering, confidence culling, and per-pair PLY export.
 *
 * These previously lived as inline lambdas inside IcpFusion's main() or were
 * duplicated verbatim across the benchmark translation units; collecting them here
 * gives a single, testable source of truth.
 */
namespace IcpUtils
{

/**
 * @brief Applies a rigid 4x4 transform to @p cloud in place: every point is mapped
 *        by (R * p + t) and every normal is rotated by R. The transform is assumed
 *        rigid, so rotating the normals preserves their unit length.
 */
void applyRigid(PointCloud &cloud, const Eigen::Matrix4f &transform);

/**
 * @brief Builds a rigid transform of @p angleDeg about a random axis combined with
 *        @p transMag along a random direction (both Gaussian-sampled then normalized).
 *        Used by the benchmark / verification harnesses to generate a known
 *        perturbation that ICP is then asked to recover.
 */
Eigen::Matrix4f randomRigid(double angleDeg, double transMag, std::mt19937 &rng);

/**
 * @brief Moves @p cloud from the pair's RECTIFIED left-camera frame into the DTU
 *        world frame, in place (points and normals):
 *
 *          x_cam   = R1^T * x_rect        (undo the rectification rotation)
 *          x_world = R^T  * x_cam + C     (DTU pose: x_cam = R * (x_world - C))
 *
 *        With every cloud placed in the shared world frame the clouds are
 *        co-registered by calibration, leaving ICP to correct only residual error.
 *
 * @param R1       Rectification rotation for the left view (res.R1, 3x3 CV_64F).
 * @param poseLeft DTU pose of the left view (R world->cam, t = camera centre C).
 */
void transformCloudToWorld(PointCloud &cloud, const cv::Mat &R1, const CameraPose &poseLeft);

/**
 * @brief Orders a candidate DTU view pair by its baseline direction in the LEFT
 *        camera frame so the dense stereo stack always sees a horizontal,
 *        left-to-right baseline.
 *
 * The DTU cameras follow a snake path: each row is swept left/right (horizontal
 * baseline) with a vertical transition between rows. The dense stereo stack
 * (SGBM, disparity-range estimation, triangulation) only handles a HORIZONTAL,
 * left-to-right baseline, so:
 *   - vertical baseline          -> unusable by this pipeline            -> return false (skip)
 *   - horizontal, cam b on +x    -> already left-to-right                -> keep order
 *   - horizontal, cam b on -x    -> reversed L/R (negative disparity)    -> swap the images
 *
 * @return false if the pair is vertical (skip); otherwise fills the ordered
 *         (@p leftView, @p rightView) and their matching (@p poseLeft, @p poseRight).
 */
bool orderPair(DTULoader &loader, int a, int b,
               int &leftView, int &rightView,
               CameraPose &poseLeft, CameraPose &poseRight);

/**
 * @brief Confidence cull: drops points whose per-point weight is below @p keepFrac
 *        of the cloud's maximum weight, keeping every per-point attribute array
 *        (colors / normals / validNormal / weights) in lockstep. Weights are
 *        frame-invariant, so this may run in any coordinate frame.
 * @return number of points removed (0 if disabled via keepFrac <= 0 or inapplicable).
 */
size_t cullByConfidence(PointCloud &cloud, float keepFrac);

/**
 * @brief Writes a single per-pair contribution to pointcloud_pair_LL_RR<suffix>.ply
 *        in the metric frame (denormalized with @p mean / @p scale). The cloud is
 *        copied before denormalizing, so the caller's normalized cloud is left
 *        untouched.
 */
void saveIndividualCloud(const PointCloud &cloud, int leftView, int rightView,
                         const Eigen::Vector3f &mean, float scale,
                         const std::string &suffix = "");

} // namespace IcpUtils
