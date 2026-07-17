#include "ICP.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// ============================================================================
// Optimization constraints
//
// Pose is parameterized as 6 doubles: [0,1,2] angle-axis rotation vector,
// [3,4,5] translation. Ceres squares the residual, so a per-correspondence
// confidence weight w enters as sqrt(w): [sqrt(w)*e]^2 == w*e^2.
// ============================================================================

namespace {

// Applies the 6-DOF pose (angle-axis + translation) to a source point.
template <typename T>
void applyPose(const T* const pose, const T* const srcPoint, T* dstPoint)
{
    ceres::AngleAxisRotatePoint(pose, srcPoint, dstPoint); // Rodrigues formula
    dstPoint[0] += pose[3];
    dstPoint[1] += pose[4];
    dstPoint[2] += pose[5];
}

// Converts an optimized 6-DOF pose vector into a 4x4 homogeneous transform.
Eigen::Matrix4f poseVectorToMatrix(const double* pose)
{
    double R[9];
    ceres::AngleAxisToRotationMatrix(pose, R); // column-major

    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) = (float)R[0]; m(0, 1) = (float)R[3]; m(0, 2) = (float)R[6];
    m(1, 0) = (float)R[1]; m(1, 1) = (float)R[4]; m(1, 2) = (float)R[7];
    m(2, 0) = (float)R[2]; m(2, 1) = (float)R[5]; m(2, 2) = (float)R[8];
    m(0, 3) = (float)pose[3];
    m(1, 3) = (float)pose[4];
    m(2, 3) = (float)pose[5];
    return m;
}

} // namespace

/**
 * Confidence-weighted point-to-point residual (3D).
 */
class PointToPointConstraint
{
public:
    PointToPointConstraint(const Eigen::Vector3f& sourcePoint, const Eigen::Vector3f& targetPoint, float weight)
        : m_sourcePoint{ sourcePoint.cast<double>() },
          m_targetPoint{ targetPoint.cast<double>() },
          m_sqrtWeight{ std::sqrt(std::max(0.0, (double)weight)) }
    { }

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const
    {
        T src[3] = { T(m_sourcePoint[0]), T(m_sourcePoint[1]), T(m_sourcePoint[2]) };
        T transformed[3];
        applyPose(pose, src, transformed);

        const T w = T(m_sqrtWeight);
        residuals[0] = w * (transformed[0] - T(m_targetPoint[0]));
        residuals[1] = w * (transformed[1] - T(m_targetPoint[1]));
        residuals[2] = w * (transformed[2] - T(m_targetPoint[2]));
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Vector3f& sourcePoint, const Eigen::Vector3f& targetPoint, float weight)
    {
        return new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
            new PointToPointConstraint(sourcePoint, targetPoint, weight));
    }

protected:
    const Eigen::Vector3d m_sourcePoint;
    const Eigen::Vector3d m_targetPoint;
    const double          m_sqrtWeight;
};

/**
 * Confidence-weighted point-to-plane residual (1D, along the target normal).
 */
class PointToPlaneConstraint
{
public:
    PointToPlaneConstraint(const Eigen::Vector3f& sourcePoint, const Eigen::Vector3f& targetPoint,
                           const Eigen::Vector3f& targetNormal, float weight)
        : m_sourcePoint{ sourcePoint.cast<double>() },
          m_targetPoint{ targetPoint.cast<double>() },
          m_targetNormal{ targetNormal.cast<double>() },
          m_sqrtWeight{ std::sqrt(std::max(0.0, (double)weight)) }
    { }

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const
    {
        T src[3] = { T(m_sourcePoint[0]), T(m_sourcePoint[1]), T(m_sourcePoint[2]) };
        T transformed[3];
        applyPose(pose, src, transformed);

        const T w = T(m_sqrtWeight);
        residuals[0] = w * (T(m_targetNormal[0]) * (transformed[0] - T(m_targetPoint[0])) +
                            T(m_targetNormal[1]) * (transformed[1] - T(m_targetPoint[1])) +
                            T(m_targetNormal[2]) * (transformed[2] - T(m_targetPoint[2])));
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Vector3f& sourcePoint, const Eigen::Vector3f& targetPoint,
                                       const Eigen::Vector3f& targetNormal, float weight)
    {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
            new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, weight));
    }

protected:
    const Eigen::Vector3d m_sourcePoint;
    const Eigen::Vector3d m_targetPoint;
    const Eigen::Vector3d m_targetNormal;
    const double          m_sqrtWeight;
};

// ============================================================================
// ICPOptimizer (base): configuration + shared geometry helpers
// ============================================================================

ICPOptimizer::ICPOptimizer()
    : m_usePointToPlane{ false },
      m_useWeights{ true },
      m_verbose{ true },
      m_nIterations{ 30 },
      m_lastMatchCount{ 0 }
{
    m_nearestNeighborSearch.setMatchingMaxDistance(0.1f);
}

void ICPOptimizer::setMatchingMaxDistance(float maxDistance) { m_nearestNeighborSearch.setMatchingMaxDistance(maxDistance); }
void ICPOptimizer::setNbOfIterations(unsigned nIterations)   { m_nIterations = nIterations; }
void ICPOptimizer::usePointToPlaneConstraints(bool enable)   { m_usePointToPlane = enable; }
void ICPOptimizer::useWeights(bool enable)                   { m_useWeights = enable; }
void ICPOptimizer::setVerbose(bool enable)                   { m_verbose = enable; }

std::vector<Eigen::Vector3f> ICPOptimizer::transformPoints(
    const std::vector<Eigen::Vector3f>& points, const Eigen::Matrix4f& pose) const
{
    const Eigen::Matrix3f R = pose.block<3, 3>(0, 0);
    const Eigen::Vector3f t = pose.block<3, 1>(0, 3);
    std::vector<Eigen::Vector3f> out;
    out.reserve(points.size());
    for (const auto& p : points)
        out.push_back(R * p + t);
    return out;
}

std::vector<Eigen::Vector3f> ICPOptimizer::transformNormals(
    const std::vector<Eigen::Vector3f>& normals, const Eigen::Matrix4f& pose) const
{
    // Pose is rigid, so the rotation block transforms normals directly.
    const Eigen::Matrix3f R = pose.block<3, 3>(0, 0);
    std::vector<Eigen::Vector3f> out;
    out.reserve(normals.size());
    for (const auto& n : normals)
        out.push_back(R * n);
    return out;
}

void ICPOptimizer::pruneCorrespondences(
    const std::vector<Eigen::Vector3f>& sourceNormals,
    const std::vector<Eigen::Vector3f>& targetNormals,
    const std::vector<bool>& targetValidNormal,
    std::vector<Match>& matches) const
{
    // Requires per-point normals on both sides to be meaningful.
    if (sourceNormals.size() != matches.size() || targetNormals.empty())
        return;

    for (size_t i = 0; i < matches.size(); ++i)
    {
        Match& match = matches[i];
        if (match.idx < 0) continue;
        if ((size_t)match.idx >= targetNormals.size()) continue;
        if (!targetValidNormal.empty() && !targetValidNormal[match.idx]) continue;

        const Eigen::Vector3f& sn = sourceNormals[i];
        const Eigen::Vector3f& tn = targetNormals[match.idx];
        if (!sn.allFinite() || !tn.allFinite()) continue;
        // Invalid source normals are stored as zero vectors; they carry no
        // orientation evidence, so the match must survive (it falls back to a
        // point-to-point residual) instead of auto-failing the dot test.
        if (sn.squaredNorm() < 1e-12f || tn.squaredNorm() < 1e-12f) continue;

        // Reject correspondences whose normals disagree by more than 60 degrees.
        if (sn.dot(tn) < 0.5f)
            match.idx = -1;
    }
}

// ============================================================================
// CeresICPOptimizer
// ============================================================================

namespace {

void configureSolver(ceres::Solver::Options& options)
{
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps     = false;
    options.linear_solver_type         = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations         = 10; // low per outer step; the KD-tree re-associates each iteration
    options.num_threads                = 8;
}

} // namespace

Eigen::Matrix4f CeresICPOptimizer::estimatePose(
    const PointCloud& source, const PointCloud& target, const Eigen::Matrix4f& initialPose)
{
    m_lastMatchCount = 0;
    m_nearestNeighborSearch.buildIndex(target.pts);

    const bool haveSourceNormals = source.normals.size() == source.pts.size();
    Eigen::Matrix4f estimatedPose = initialPose;

    Eigen::Matrix4f bestPose = initialPose;
    double bestMeanDist = std::numeric_limits<double>::max();
    int bestMatchCount = 0;
    int divergingStreak = 0;


    // Outlier Rejection Lambda: transforms the source by the current pose, queries nearest neighbors, and prunes matches.
    auto associate = [&](const Eigen::Matrix4f& pose,
                         const std::vector<Eigen::Vector3f>& transformedPoints,
                         int& matched, double& meanDist)
    {
        auto matches = m_nearestNeighborSearch.queryMatches(transformedPoints);

        std::vector<float> validSq;
        validSq.reserve(matches.size());
        for (const auto& mm : matches)
            if (mm.idx >= 0)
                validSq.push_back(mm.distance);
        if (!validSq.empty())
        {
            const size_t mid = validSq.size() / 2;
            std::nth_element(validSq.begin(), validSq.begin() + mid, validSq.end());
            // 3x median distance == 9x median squared distance; the floor keeps
            // the gate open once the clouds sit within sampling noise.
            const float tauSq = std::max(9.0f * validSq[mid], 1e-8f);
            for (auto& mm : matches)
                if (mm.idx >= 0 && mm.distance > tauSq)
                    mm.idx = -1;
        }

        if (m_usePointToPlane && haveSourceNormals)
        {
            const auto transformedNormals = transformNormals(source.normals, pose);
            pruneCorrespondences(transformedNormals, target.normals, target.validNormal, matches);
        }

        matched = 0;
        double sumDist = 0.0;
        for (const auto& mm : matches)
        {
            if (mm.idx < 0) continue;
            ++matched;
            sumDist += std::sqrt((double)mm.distance);
        }
        meanDist = matched > 0 ? sumDist / matched : std::numeric_limits<double>::max();
        return matches;
    };

    for (unsigned iter = 0; iter < m_nIterations; ++iter)
    {
        // Associate: transform source by the current estimate, then match
        const auto transformedPoints = transformPoints(source.pts, estimatedPose);
        int matched = 0;
        double meanDist = 0.0;
        auto matches = associate(estimatedPose, transformedPoints, matched, meanDist);
        m_lastMatchCount = matched;

        if (matched < 6)
        {
            // Report actual nearest-neighbor spread so "just missed" is distinguishable
            // from "wildly misaligned".
            float minD = std::numeric_limits<float>::max(), maxD = 0.0f;
            double sumD = 0.0;
            for (const auto& mm : matches) { minD = std::min(minD, mm.distance); maxD = std::max(maxD, mm.distance); sumD += mm.distance; }
            const double meanD = matches.empty() ? 0.0 : sumD / matches.size();
            std::cout << "  [ICP iter " << iter << "] only " << matched
                      << " correspondences survived -- stopping (need >= 6).\n"
                      << "    NN distances (all " << matches.size() << " source pts): min="
                      << std::sqrt(minD) << " rms=" << std::sqrt(meanD)
                      << " max=" << std::sqrt(maxD) << "\n";
            break;
        }

        if (meanDist < bestMeanDist)
        {
            bestMeanDist = meanDist;
            bestPose = estimatedPose;
            bestMatchCount = matched;
            divergingStreak = 0;
        }
        // Early stopping if no divergence is observed
        else if (++divergingStreak >= 5)
        {
            if (m_verbose)
                std::cout << "  [ICP iter " << iter << "] mean distance has not improved for "
                          << divergingStreak << " iterations (best " << bestMeanDist
                          << ", now " << meanDist << ") -- stopping.\n";
            break;
        }

        // --- Diagnostics ---
        if (m_verbose)
        {
            int planeCount = 0;
            for (size_t i = 0; i < matches.size(); ++i)
            {
                if (matches[i].idx < 0) continue;
                if (m_usePointToPlane && matches[i].idx < (int)target.normals.size()
                    && (target.validNormal.empty() || target.validNormal[matches[i].idx]))
                    ++planeCount;
            }
            std::cout << "  [ICP iter " << iter << "] correspondences=" << matched
                      << "/" << source.pts.size()
                      << " | plane=" << planeCount << " point=" << (matched - planeCount)
                      << " | mean pre-opt dist=" << meanDist << "\n";
        }

        // --- Solve for the incremental pose (starts at identity each iteration) ---
        double poseIncrement[6] = { 0, 0, 0, 0, 0, 0 };

        ceres::Problem problem;
        for (size_t i = 0; i < matches.size(); ++i)
        {
            const Match& match = matches[i];
            if (match.idx < 0) continue;

            const Eigen::Vector3f& sp = transformedPoints[i];
            const Eigen::Vector3f& tp = target.pts[match.idx];
            if (!sp.allFinite() || !tp.allFinite()) continue;

            const float weight = (m_useWeights && i < source.weights.size()) ? source.weights[i] : 1.0f;

            const bool usePlane = m_usePointToPlane
                                && match.idx < (int)target.normals.size()
                                && (target.validNormal.empty() || target.validNormal[match.idx])
                                && target.normals[match.idx].allFinite();

            if (usePlane)
                problem.AddResidualBlock(
                    PointToPlaneConstraint::create(sp, tp, target.normals[match.idx], weight),
                    nullptr, poseIncrement);
            else
                problem.AddResidualBlock(
                    PointToPointConstraint::create(sp, tp, weight),
                    nullptr, poseIncrement);
        }

        ceres::Solver::Options options;
        configureSolver(options);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (m_verbose)
            std::cout << "    Ceres: " << summary.BriefReport() << "\n";

        // Accumulate from the left (left-increment notation).
        estimatedPose = poseVectorToMatrix(poseIncrement) * estimatedPose;

        const double omegaNorm = std::sqrt(poseIncrement[0] * poseIncrement[0] +
                                           poseIncrement[1] * poseIncrement[1] +
                                           poseIncrement[2] * poseIncrement[2]);
        const double transNorm = std::sqrt(poseIncrement[3] * poseIncrement[3] +
                                           poseIncrement[4] * poseIncrement[4] +
                                           poseIncrement[5] * poseIncrement[5]);
        if (m_verbose)
            std::cout << "    iter " << iter << " step: |trans|=" << transNorm
                      << " |omega|=" << omegaNorm << "\n";

        if (transNorm < 1e-4 && omegaNorm < 1e-4)
        {
            if (m_verbose) std::cout << "  [ICP] converged at iter " << iter << "\n";
            break;
        }
    }

    {
        const auto transformedPoints = transformPoints(source.pts, estimatedPose);
        int matched = 0;
        double meanDist = 0.0;
        associate(estimatedPose, transformedPoints, matched, meanDist);
        if (matched >= 6 && meanDist < bestMeanDist)
        {
            bestMeanDist = meanDist;
            bestPose = estimatedPose;
            bestMatchCount = matched;
        }
    }
    m_lastMatchCount = bestMatchCount;

    if (m_verbose)
        std::cout << "  [ICP] returning best pose (mean correspondence dist "
                  << bestMeanDist << ", " << m_lastMatchCount << " matches).\n";

    return bestPose;
}
