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
      m_lastMatchCount{ 0 },
      m_maxDistance{ 0.1f },
      m_robustLoss{ ICPRobustLoss::None },
      m_robustScale{ 0.02 },
      m_useReciprocal{ false },
      m_useAdaptiveGate{ false },
      m_trimFraction{ 1.0f }
{
    m_nearestNeighborSearch.setMatchingMaxDistance(0.1f);
}

void ICPOptimizer::setMatchingMaxDistance(float maxDistance) { m_maxDistance = maxDistance; m_nearestNeighborSearch.setMatchingMaxDistance(maxDistance); }
void ICPOptimizer::setNbOfIterations(unsigned nIterations)   { m_nIterations = nIterations; }
void ICPOptimizer::usePointToPlaneConstraints(bool enable)   { m_usePointToPlane = enable; }
void ICPOptimizer::useWeights(bool enable)                   { m_useWeights = enable; }
void ICPOptimizer::setVerbose(bool enable)                   { m_verbose = enable; }
void ICPOptimizer::setRobustLoss(ICPRobustLoss loss, double scale)
{
    m_robustLoss = loss;
    m_robustScale = std::max(scale, 1e-8);
}
void ICPOptimizer::useReciprocalCorrespondences(bool enable) { m_useReciprocal = enable; }
void ICPOptimizer::setTrimFraction(float fraction)            { m_trimFraction = std::clamp(fraction, 0.05f, 1.0f); }
void ICPOptimizer::useAdaptiveDistanceGate(bool enable)       { m_useAdaptiveGate = enable; }

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
        if (sn.squaredNorm() < 1e-12f || tn.squaredNorm() < 1e-12f) continue;

        // Reject correspondences whose normals disagree by more than 60 degrees.
        if (sn.dot(tn) < 0.5f)
            match.idx = -1;
    }
}

namespace {

double medianOf(std::vector<double> values)
{
    if (values.empty()) return std::numeric_limits<double>::infinity();
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double m = values[mid];
    if (values.size() % 2 == 0)
    {
        auto lo = std::max_element(values.begin(), values.begin() + mid);
        m = 0.5 * (m + *lo);
    }
    return m;
}

} // namespace

void ICPOptimizer::robustlyFilterCorrespondences(
    const std::vector<Eigen::Vector3f>& transformedPoints,
    const PointCloud& target,
    std::vector<Match>& matches) const
{
    if (m_useReciprocal && !target.pts.empty() && !transformedPoints.empty())
    {
        NearestNeighborSearch reverse;
        reverse.setMatchingMaxDistance(m_maxDistance);
        reverse.buildIndex(transformedPoints);
        const auto reverseMatches = reverse.queryMatches(target.pts);
        for (size_t i = 0; i < matches.size(); ++i)
        {
            const int j = matches[i].idx;
            if (j >= 0 && (j >= static_cast<int>(reverseMatches.size()) || reverseMatches[j].idx != static_cast<int>(i)))
                matches[i].idx = -1;
        }
    }

    std::vector<double> distances;
    distances.reserve(matches.size());
    for (const Match& m : matches)
        if (m.idx >= 0) distances.push_back(std::sqrt(std::max(0.0f, m.distance)));
    if (distances.empty()) return;

    if (m_useAdaptiveGate && distances.size() >= 10)
    {
        const double med = medianOf(distances);
        std::vector<double> deviations;
        deviations.reserve(distances.size());
        for (double d : distances) deviations.push_back(std::abs(d - med));
        const double sigma = 1.4826 * medianOf(deviations);
        const double adaptive = std::min<double>(m_maxDistance,
            std::max<double>(0.25 * m_maxDistance, med + 3.0 * std::max(sigma, 1e-6)));
        for (Match& m : matches)
            if (m.idx >= 0 && std::sqrt(std::max(0.0f, m.distance)) > adaptive) m.idx = -1;
    }

    if (m_trimFraction < 0.999f)
    {
        std::vector<float> validSq;
        validSq.reserve(matches.size());
        for (const Match& m : matches) if (m.idx >= 0) validSq.push_back(m.distance);
        if (validSq.size() >= 10)
        {
            const size_t keep = std::max<size_t>(6, static_cast<size_t>(std::ceil(m_trimFraction * validSq.size())));
            const size_t kth = std::min(keep, validSq.size()) - 1;
            std::nth_element(validSq.begin(), validSq.begin() + kth, validSq.end());
            const float cutoff = validSq[kth];
            for (Match& m : matches)
                if (m.idx >= 0 && m.distance > cutoff) m.idx = -1;
        }
    }
}

ICPMetrics ICPOptimizer::computeMetrics(
    const PointCloud& source, const PointCloud& target,
    const std::vector<Eigen::Vector3f>& transformedPoints,
    const std::vector<Match>& matches) const
{
    ICPMetrics out;
    out.sourceCount = static_cast<int>(source.pts.size());
    std::vector<double> distances;
    distances.reserve(matches.size());
    double sum = 0.0, sumSq = 0.0;
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();

    Eigen::Vector3f mn = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f mx = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    for (const auto& p : transformedPoints) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    const Eigen::Vector3f center = 0.5f * (mn + mx);
    unsigned sourceMask = 0, matchedMask = 0;

    for (size_t i = 0; i < transformedPoints.size(); ++i)
    {
        const Eigen::Vector3f& p = transformedPoints[i];
        const unsigned oct = (p.x() >= center.x() ? 1u : 0u) |
                             (p.y() >= center.y() ? 2u : 0u) |
                             (p.z() >= center.z() ? 4u : 0u);
        sourceMask |= 1u << oct;
        if (i >= matches.size() || matches[i].idx < 0) continue;
        const int j = matches[i].idx;
        if (j >= static_cast<int>(target.pts.size())) continue;
        matchedMask |= 1u << oct;
        const double d = (p - target.pts[j]).norm();
        distances.push_back(d); sum += d; sumSq += d * d;

        const double w = (m_useWeights && i < source.weights.size())
                       ? std::max<double>(source.weights[i], 0.0) : 1.0;
        if (m_usePointToPlane && j < static_cast<int>(target.normals.size()) &&
            (target.validNormal.empty() || target.validNormal[j]) &&
            target.normals[j].squaredNorm() > 1e-12f)
        {
            const Eigen::Vector3d pd = p.cast<double>();
            const Eigen::Vector3d n = target.normals[j].cast<double>().normalized();
            Eigen::Matrix<double, 1, 6> J;
            J << pd.cross(n).transpose(), n.transpose();
            H.noalias() += w * J.transpose() * J;
        }
        else
        {
            Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
            Eigen::Matrix3d skew;
            const Eigen::Vector3d pd = p.cast<double>();
            skew << 0.0, -pd.z(), pd.y(), pd.z(), 0.0, -pd.x(), -pd.y(), pd.x(), 0.0;
            J.block<3,3>(0,0) = -skew;
            J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
            H.noalias() += w * J.transpose() * J;
        }
    }

    out.matchCount = static_cast<int>(distances.size());
    out.overlap = out.sourceCount ? static_cast<double>(out.matchCount) / out.sourceCount : 0.0;
    if (!distances.empty())
    {
        out.meanDistance = sum / distances.size();
        out.rmse = std::sqrt(sumSq / distances.size());
        out.medianDistance = medianOf(distances);
    }
    out.spatialCoverage = sourceMask ? static_cast<double>(__builtin_popcount(matchedMask)) /
                                      __builtin_popcount(sourceMask) : 0.0;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(H);
    if (es.info() == Eigen::Success)
    {
        const double lo = es.eigenvalues().minCoeff();
        const double hi = es.eigenvalues().maxCoeff();
        if (lo > 1e-12) out.conditionNumber = hi / lo;
    }
    return out;
}

ICPMetrics ICPOptimizer::evaluatePose(const PointCloud& source, const PointCloud& target,
                                      const Eigen::Matrix4f& pose)
{
    m_nearestNeighborSearch.buildIndex(target.pts);
    const auto transformed = transformPoints(source.pts, pose);
    auto matches = m_nearestNeighborSearch.queryMatches(transformed);
    if (m_usePointToPlane && source.normals.size() == source.pts.size())
        pruneCorrespondences(transformNormals(source.normals, pose), target.normals,
                             target.validNormal, matches);
    robustlyFilterCorrespondences(transformed, target, matches);
    return computeMetrics(source, target, transformed, matches);
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
    m_initialMetrics = evaluatePose(source, target, initialPose);
    // evaluatePose rebuilds the same target index; make this explicit for clarity.
    m_nearestNeighborSearch.buildIndex(target.pts);

    for (unsigned iter = 0; iter < m_nIterations; ++iter)
    {
        // --- Associate: transform source by the current estimate, then match ---
        const auto transformedPoints = transformPoints(source.pts, estimatedPose);
        auto matches = m_nearestNeighborSearch.queryMatches(transformedPoints);

        if (m_usePointToPlane && haveSourceNormals)
        {
            const auto transformedNormals = transformNormals(source.normals, estimatedPose);
            pruneCorrespondences(transformedNormals, target.normals, target.validNormal, matches);
        }
        robustlyFilterCorrespondences(transformedPoints, target, matches);

        const int matched = (int)std::count_if(matches.begin(), matches.end(),
                                                [](const Match& m) { return m.idx >= 0; });
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
                      << std::sqrt(minD) << " mean=" << std::sqrt(meanD)
                      << " max=" << std::sqrt(maxD) << "\n";
            break;
        }

        // --- Diagnostics ---
        if (m_verbose)
        {
            double sumDist = 0.0;
            int planeCount = 0;
            for (size_t i = 0; i < matches.size(); ++i)
            {
                if (matches[i].idx < 0) continue;
                sumDist += (transformedPoints[i] - target.pts[matches[i].idx]).norm();
                if (m_usePointToPlane && matches[i].idx < (int)target.normals.size()
                    && (target.validNormal.empty() || target.validNormal[matches[i].idx]))
                    ++planeCount;
            }
            std::cout << "  [ICP iter " << iter << "] correspondences=" << matched
                      << "/" << source.pts.size()
                      << " | plane=" << planeCount << " point=" << (matched - planeCount)
                      << " | mean pre-opt dist=" << (sumDist / matched) << "\n";
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
            {
                ceres::LossFunction* loss = nullptr;
                if (m_robustLoss == ICPRobustLoss::Huber) loss = new ceres::HuberLoss(m_robustScale);
                else if (m_robustLoss == ICPRobustLoss::Cauchy) loss = new ceres::CauchyLoss(m_robustScale);
                problem.AddResidualBlock(
                    PointToPlaneConstraint::create(sp, tp, target.normals[match.idx], weight),
                    loss, poseIncrement);
            }
            else
            {
                ceres::LossFunction* loss = nullptr;
                if (m_robustLoss == ICPRobustLoss::Huber) loss = new ceres::HuberLoss(m_robustScale);
                else if (m_robustLoss == ICPRobustLoss::Cauchy) loss = new ceres::CauchyLoss(m_robustScale);
                problem.AddResidualBlock(
                    PointToPointConstraint::create(sp, tp, weight),
                    loss, poseIncrement);
            }
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

        if (transNorm < 1e-5 && omegaNorm < 1e-5)
        {
            if (m_verbose) std::cout << "  [ICP] converged at iter " << iter << "\n";
            break;
        }
    }

    // Re-associate after the final increment. This makes the public match count and
    // quality metrics describe the returned pose rather than the pre-update state.
    m_finalMetrics = evaluatePose(source, target, estimatedPose);
    m_lastMatchCount = m_finalMetrics.matchCount;

    return estimatedPose;
}
