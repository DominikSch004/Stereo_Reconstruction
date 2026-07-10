// Synthetic verification for ICP::align.
//
// Verifies, against a known ground-truth rigid transform, that:
//   A. (exact recovery)   noise-free clouds -> all 4 variants recover the
//      perturbation to sub-degree / sub-1% accuracy
//   B. (weight sanity)    homoscedastic noise + uniform weights -> weighted and
//      unweighted results are statistically indistinguishable
//   C. (weight benefit)   heteroscedastic noise with weights = 1/sigma^2 ->
//      weighted ICP recovers the transform strictly better than unweighted
//
// The surface z = 0.3*sin(2x)*cos(2y) has analytic normals, so point-to-plane
// is tested against exact normals rather than estimated ones.
//
// Exit code 0 iff all checks pass.

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "PlyUtils.hpp"
#include "ICP.hpp"
#include "IcpUtils.hpp"

namespace {

using IcpUtils::applyRigid;
using IcpUtils::randomRigid;

PointCloud makeSurface(int nGrid, double jitter, std::mt19937 &rng)
{
    PointCloud cloud;
    std::uniform_real_distribution<double> jit(-jitter, jitter);
    for (int iy = 0; iy < nGrid; ++iy)
    {
        for (int ix = 0; ix < nGrid; ++ix)
        {
            double x = -1.0 + 2.0 * ix / (nGrid - 1) + jit(rng);
            double y = -1.0 + 2.0 * iy / (nGrid - 1) + jit(rng);
            double z = 0.3 * std::sin(2.0 * x) * std::cos(2.0 * y);

            double dzdx = 0.6 * std::cos(2.0 * x) * std::cos(2.0 * y);
            double dzdy = -0.6 * std::sin(2.0 * x) * std::sin(2.0 * y);
            Eigen::Vector3f n(-dzdx, -dzdy, 1.0);
            n.normalize();

            cloud.pts.push_back(Eigen::Vector3f((float)x, (float)y, (float)z));
            cloud.colors.push_back(cv::Vec3b(200, 200, 200));
            cloud.weights.push_back(1.0f);
            cloud.normals.push_back(n);
            cloud.validNormal.push_back(true);
        }
    }
    return cloud;
}

struct TrialResult
{
    double rotErrDeg;  // rotation angle of T_recovered * T_perturbation (identity if perfect)
    double transErr;   // translation norm of the same composition
    double rmse;       // aligned points vs their pre-perturbation positions
};

TrialResult runVariant(const PointCloud &srcPerturbed,
                       const std::vector<Eigen::Vector3f> &srcRefPts,
                       const PointCloud &target,
                       const Eigen::Matrix4f &T_pert,
                       bool useWeights, ICPMode mode,
                       int maxIter, float distThresh)
{
    PointCloud src = srcPerturbed;
    CeresICPOptimizer icp;
    icp.setMode(mode);
    icp.useWeights(useWeights);
    icp.setNbOfIterations(maxIter);
    icp.setMatchingMaxDistance(distThresh);
    icp.setVerbose(false);
    Eigen::Matrix4f T_rec = icp.estimatePose(src, target);

    // estimatePose does not mutate the source; materialize the aligned points for the RMSE below.
    applyRigid(src, T_rec);

    Eigen::Matrix4f E = T_rec * T_pert; // perfect recovery -> identity
    Eigen::Matrix3f R_err = E.block<3, 3>(0, 0);
    double c = std::min(1.0, std::max(-1.0, (double)((R_err.trace() - 1.0f) / 2.0f)));
    TrialResult r;
    r.rotErrDeg = std::acos(c) * 180.0 / M_PI;
    r.transErr = E.block<3, 1>(0, 3).norm();

    double sq = 0.0;
    for (size_t i = 0; i < src.pts.size(); ++i)
        sq += (src.pts[i] - srcRefPts[i]).squaredNorm();
    r.rmse = std::sqrt(sq / src.pts.size());
    return r;
}

const char *variantName(bool w, ICPMode m)
{
    if (m == ICPMode::PointToPoint)
        return w ? "P2Point weighted  " : "P2Point unweighted";
    return w ? "P2Plane weighted  " : "P2Plane unweighted";
}

} // namespace

int main()
{
    std::mt19937 rng(1234);
    const int maxIter = 50;
    const float distThresh = 0.3f;
    const double angleDeg = 5.0, transMag = 0.1;

    // Target and source sample the surface at different (jittered) locations so
    // correspondences are surface-to-surface, not trivially index-to-index.
    PointCloud target = makeSurface(80, 0.004, rng);
    PointCloud srcClean = makeSurface(75, 0.004, rng);

    const Eigen::Matrix4f T_pert = randomRigid(angleDeg, transMag, rng);

    struct Variant { bool w; ICPMode m; };
    const std::vector<Variant> variants = {
        {false, ICPMode::PointToPoint}, {true, ICPMode::PointToPoint},
        {false, ICPMode::PointToPlane}, {true, ICPMode::PointToPlane}};

    bool allPass = true;
    auto check = [&](bool cond, const std::string &what) {
        std::cout << (cond ? "  [PASS] " : "  [FAIL] ") << what << "\n";
        if (!cond) allPass = false;
    };

    std::cout << std::fixed << std::setprecision(5);

    // ---------------- Case A: noise-free exact recovery ----------------
    std::cout << "\n=== Case A: noise-free, perturbation " << angleDeg
              << " deg / " << transMag << " ===\n";
    {
        PointCloud srcPert = srcClean;
        applyRigid(srcPert, T_pert);
        for (const auto &v : variants)
        {
            TrialResult r = runVariant(srcPert, srcClean.pts, target, T_pert,
                                       v.w, v.m, maxIter, distThresh);
            std::cout << variantName(v.w, v.m) << ": rotErr=" << r.rotErrDeg
                      << " deg, transErr=" << r.transErr << ", rmse=" << r.rmse << "\n";
            check(r.rotErrDeg < 0.5 && r.transErr < 0.02 && r.rmse < 0.02,
                  std::string(variantName(v.w, v.m)) + " exact recovery");
        }
    }

    // ---------------- Case B: homoscedastic noise, uniform weights ----------------
    std::cout << "\n=== Case B: homoscedastic noise sigma=0.01, uniform weights ===\n";
    {
        PointCloud srcNoisy = srcClean;
        std::normal_distribution<float> g(0.0f, 0.01f);
        for (auto &p : srcNoisy.pts)
            p += Eigen::Vector3f(g(rng), g(rng), g(rng));
        std::vector<Eigen::Vector3f> refPts = srcNoisy.pts;
        applyRigid(srcNoisy, T_pert);

        for (auto mode : {ICPMode::PointToPoint, ICPMode::PointToPlane})
        {
            TrialResult ru = runVariant(srcNoisy, refPts, target, T_pert, false, mode, maxIter, distThresh);
            TrialResult rw = runVariant(srcNoisy, refPts, target, T_pert, true, mode, maxIter, distThresh);
            std::cout << variantName(false, mode) << ": rotErr=" << ru.rotErrDeg
                      << " rmse=" << ru.rmse << "\n";
            std::cout << variantName(true, mode) << ": rotErr=" << rw.rotErrDeg
                      << " rmse=" << rw.rmse << "\n";
            check(ru.rotErrDeg < 1.0 && rw.rotErrDeg < 1.0,
                  "both variants converge under uniform noise");
            check(std::abs(ru.rmse - rw.rmse) < 0.2 * std::max(ru.rmse, 1e-6),
                  "uniform weights leave the result unchanged (<20% rmse difference)");
        }
    }

    // ---------------- Case C: heteroscedastic noise, informative weights ----------------
    std::cout << "\n=== Case C: heteroscedastic noise (70% sigma=0.005, 30% sigma=0.08), "
                 "weights = 1/sigma^2 ===\n";
    {
        PointCloud srcNoisy = srcClean;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (size_t i = 0; i < srcNoisy.pts.size(); ++i)
        {
            const bool corrupted = u(rng) < 0.3;
            const float sigma = corrupted ? 0.08f : 0.005f;
            std::normal_distribution<float> g(0.0f, sigma);
            srcNoisy.pts[i] += Eigen::Vector3f(g(rng), g(rng), g(rng));
            srcNoisy.weights[i] = 1.0f / (sigma * sigma);
        }
        std::vector<Eigen::Vector3f> refPts = srcNoisy.pts;
        applyRigid(srcNoisy, T_pert);

        for (auto mode : {ICPMode::PointToPoint, ICPMode::PointToPlane})
        {
            TrialResult ru = runVariant(srcNoisy, refPts, target, T_pert, false, mode, maxIter, distThresh);
            TrialResult rw = runVariant(srcNoisy, refPts, target, T_pert, true, mode, maxIter, distThresh);
            std::cout << variantName(false, mode) << ": rotErr=" << ru.rotErrDeg
                      << " transErr=" << ru.transErr << " rmse=" << ru.rmse << "\n";
            std::cout << variantName(true, mode) << ": rotErr=" << rw.rotErrDeg
                      << " transErr=" << rw.transErr << " rmse=" << rw.rmse << "\n";
            check(rw.rmse < ru.rmse,
                  std::string("weighted beats unweighted (") +
                      (mode == ICPMode::PointToPoint ? "P2Point" : "P2Plane") + ")");
        }
    }

    std::cout << "\n=== " << (allPass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << " ===\n";
    return allPass ? 0 : 1;
}
