#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <utility>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include "DTULoader.hpp"
#include "Pipeline.hpp"
#include "PipelineConfig.hpp"
#include "PlyUtils.hpp"
#include "ICP.hpp"
#include "IcpUtils.hpp"
#include "PoissonReconstruction.hpp"
#include "MeshUtils.hpp"
#include "VoxelFusion.hpp"

int main(int argc, char **argv)
{
    // Per-step backend selection shared with the main stereo reconstruction pipeline
    const std::string configPath = (argc > 1) ? argv[1] : "../config.yaml";
    PipelineConfig config;
    try
    {
        config = PipelineConfig::load(configPath);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
    config.print();

    // Pairs to reconstruct and fuse. Either a curated set of known-horizontal DTU
    // pairs, or an OVERLAPPING sliding window (i, i+1), (i+1, i+2), ... over the
    // configured range. Consecutive windows share a view, so their clouds share a
    // large visible surface and sit only ~one orbit step (~12 deg on DTU) apart --
    // small enough for ICP to register sequentially without the GT world transform.
    // Vertical baselines are dropped later by IcpUtils::orderPair regardless of mode;
    // on DTU those only occur at row boundaries, so keep the range inside one row to
    // avoid breaking the overlap chain.
    std::vector<std::pair<int, int>> selectedPairs;
    if (config.icpPairMode == IcpPairMode::Consecutive)
    {
        for (int v = config.icpViewFirst; v < config.icpViewLast; v += 1)
            selectedPairs.emplace_back(v, v + 1);
        std::cout << "Pair source: overlapping sliding window over views " << config.icpViewFirst
                  << ".." << config.icpViewLast << " (" << selectedPairs.size()
                  << " candidate pairs before vertical-baseline filtering).\n";
    }
    else
    {
        selectedPairs = {
            {1, 2}, {4, 5}, {7, 8}, {12, 13}, {18, 19}, {26, 27}, {32, 33}, {37, 38}};
        std::cout << "Pair source: curated selection (" << selectedPairs.size() << " pairs).\n";
    }

    const size_t minCloudPoints = 1000; // reject degenerate reconstructions (near-empty clouds)
    const float confidenceDiscardFraction = 0.10f;
    // Measured on fusion-resolution surfels, not raw pixels. The model is much
    // sparser than an organized stereo cloud by design.
    const double minOverlapFrac = 0.05;

    std::vector<PointCloud> clouds;
    clouds.reserve(selectedPairs.size());
    // Ordered (left, right) view label for each collected cloud, index-aligned with `clouds`.
    // Used to name the individual per-pair PLY files written out below.
    std::vector<std::pair<int, int>> cloudPairs;
    cloudPairs.reserve(selectedPairs.size());
    DTULoader loader("../data/dtu/");

    for (const auto &[a, b] : selectedPairs)
    {
        int leftView = a, rightView = b;
        CameraPose poseLeft, poseRight;
        if (!IcpUtils::orderPair(loader, a, b, leftView, rightView, poseLeft, poseRight))
        {
            std::cerr << "Skipping pair (" << a << "," << b << "): vertical baseline "
                      << "(dense pipeline needs a horizontal pair).\n";
            continue;
        }

        std::cout << "\n=== Processing Pair (" << leftView << ", " << rightView << ") ["
                  << (clouds.size() + 1) << "/" << selectedPairs.size() << "] ===\n";

        StereoPair pair = loader.loadPair(leftView, rightView);
        cv::Mat K(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                K.at<double>(r, c) = poseLeft.K(r, c);

        PipelineResult res;
        // Pass both views' camera centers so Pipeline rescales t to the true metric baseline
        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, config,
                                    poseLeft.t, poseRight.t))
        {
            std::cerr << "Pipeline failed for pair (" << leftView << "," << rightView
                      << ") -- skipping.\n";
            continue;
        }
        if (std::abs(res.P2r.at<double>(0, 3)) < std::abs(res.P2r.at<double>(1, 3)))
        {
            std::cerr << "WARNING: pair (" << leftView << "," << rightView << ") rectified to a "
                      << "vertical baseline -- skipping.\n";
            continue;
        }

        PointCloud cloud = PlyUtils::buildPointCloud(
            res.denseDisparity, res.Q, res.P1r, res.P2r, res.camToWorld,
            res.rectColor, res.minDisp, res.globalConfidence,
            config.triangulation, res.disparityConfidence);

        const size_t beforeCull = cloud.pts.size();
        const size_t culled = IcpUtils::cullByConfidence(cloud, confidenceDiscardFraction);
        if (culled > 0)
            std::cout << "  Confidence cull (lowest " << (100.0f * confidenceDiscardFraction)
                      << "%): removed " << culled << " / " << beforeCull << " ("
                      << (100.0 * (double)culled / (double)beforeCull) << "%), "
                      << cloud.pts.size() << " kept.\n";

        if (cloud.pts.size() < minCloudPoints)
        {
            std::cerr << "WARNING: pair (" << leftView << "," << rightView << ") produced only "
                      << cloud.pts.size() << " points (< " << minCloudPoints
                      << ") -- degenerate reconstruction, skipping.\n";
            continue;
        }

        // EXPERIMENT: world placement via GT pose is DISABLED. Each cloud stays in its own
        // rectified left-camera frame. With disjoint pairs there is no image-based cross-pair
        // pose, so ICP (local, gated to <=10 deg) must globally register clouds separated by
        // large viewpoint rotations -- expected to fail. Restore the call below to fix.
        // IcpUtils::transformCloudToWorld(cloud, res.R1, poseLeft);

        std::cout << "Cloud from pair (" << leftView << "," << rightView << "): "
                  << cloud.pts.size() << " points generated.\n";
        clouds.push_back(std::move(cloud));
        cloudPairs.push_back({leftView, rightView});
    }

    if (clouds.empty())
    {
        std::cerr << "No point clouds were successfully generated. Aborting execution.\n";
        return -1;
    }
    if (clouds.size() < selectedPairs.size())
        std::cerr << "WARNING: collected only " << clouds.size() << " of " << selectedPairs.size()
                  << " selected pairs (some were skipped or failed to reconstruct).\n";

    // Single normalization, derived from cloud 0 only, preserving relative spatial
    // relationship between clouds. normalise() already transforms cloud 0 in place,
    // so only the remaining clouds need the same transform applied.
    auto [mean0, scale0] = PlyUtils::normalise(clouds[0]);
    for (size_t ci = 1; ci < clouds.size(); ++ci)
        for (auto &p : clouds[ci].pts)
            p = (p - mean0) / scale0;

    // The model stores one confidence-weighted surfel per local surface region.
    // Repeated observations therefore reduce noise instead of thickening the cloud.
    VoxelFusionModel fusion(config.fusionVoxelSize, config.fusionOutlierFactor);
    auto firstStats = fusion.integrate(clouds[0]);
    PointCloud fused = fusion.pointCloud();
    std::cout << "Initial fusion model: " << fused.pts.size() << " surfels ("
              << firstStats.inserted << " inserted, " << firstStats.merged << " merged).\n";
    IcpUtils::saveIndividualCloud(clouds[0], cloudPairs[0].first, cloudPairs[0].second, mean0, scale0);
    for (size_t i = 1; i < clouds.size(); ++i)
    {
        std::cout << "\nICP aligning cloud " << i + 1 << " to fused reference...\n";

        // Refine a working copy so clouds[i] retains its untouched calibration placement as fallback.
        PointCloud refined = clouds[i];

        struct Level { float voxel, maxDistance; unsigned iterations; bool weighted; };
        const std::vector<Level> levels = {
            {0.040f, 0.120f, 20, false},
            {0.020f, 0.075f, 15, true},
            {0.010f, 0.045f, 12, true}};

        Eigen::Matrix4f totalT = Eigen::Matrix4f::Identity();
        ICPMetrics lastLevelMetrics;
        for (size_t levelIdx = 0; levelIdx < levels.size(); ++levelIdx)
        {
            const Level &level = levels[levelIdx];
            PointCloud sourceLevel = PlyUtils::voxelDownsample(refined, level.voxel);
            PointCloud targetLevel = PlyUtils::voxelDownsample(fused, level.voxel);
            if (sourceLevel.pts.size() < 20 || targetLevel.pts.size() < 20) continue;

            CeresICPOptimizer icp;
            icp.setMode(config.icpMode);
            icp.setMatchingMaxDistance(level.maxDistance);
            icp.setNbOfIterations(level.iterations);
            icp.useWeights(level.weighted);
            icp.useReciprocalCorrespondences(config.icpReciprocal);
            icp.setTrimFraction(config.icpTrimFraction);
            icp.useAdaptiveDistanceGate(true);
            if (config.icpRobust)
                icp.setRobustLoss(ICPRobustLoss::Cauchy, std::max(0.5f * level.voxel, 0.003f));
            icp.setVerbose(false);

            Eigen::Matrix4f levelT = icp.estimatePose(sourceLevel, targetLevel);
            IcpUtils::applyRigid(refined, levelT);
            totalT = levelT * totalT;
            lastLevelMetrics = icp.finalMetrics();
            std::cout << "  pyramid level " << levelIdx << " voxel=" << level.voxel
                      << ": matches=" << lastLevelMetrics.matchCount
                      << ", median=" << lastLevelMetrics.medianDistance
                      << ", rmse=" << lastLevelMetrics.rmse << "\n";
        }

        // Evaluate the complete refinement against the same stable fused model.
        CeresICPOptimizer quality;
        quality.setMode(config.icpMode);
        quality.setMatchingMaxDistance(0.05f);
        quality.useWeights(true);
        quality.useReciprocalCorrespondences(config.icpReciprocal);
        quality.setTrimFraction(config.icpTrimFraction);
        quality.useAdaptiveDistanceGate(true);
        quality.setVerbose(false);
        const PointCloud calibrationEval = PlyUtils::voxelDownsample(clouds[i], config.fusionVoxelSize);
        const PointCloud refinedEval = PlyUtils::voxelDownsample(refined, config.fusionVoxelSize);
        const ICPMetrics before = quality.evaluatePose(calibrationEval, fused);
        const ICPMetrics after = quality.evaluatePose(refinedEval, fused);

        const Eigen::Matrix3f Rcorr = totalT.block<3,3>(0,0);
        const double cosAngle = std::clamp<double>((Rcorr.trace() - 1.0) * 0.5, -1.0, 1.0);
        const double correctionAngleDeg = std::acos(cosAngle) * 180.0 / M_PI;
        const double correctionTranslation = totalT.block<3,1>(0,3).norm();
        const bool improvesMedian = !before.valid() || after.medianDistance <= 0.98 * before.medianDistance;
        const bool stableRmse = !before.valid() || after.rmse <= 1.02 * before.rmse;
        const bool wellConstrained = std::isfinite(after.conditionNumber) && after.conditionNumber <= 1e12;
        // Plausibility gate (correctionAngleDeg <= 10, correctionTranslation <= 0.15) removed:
        // it assumed the GT world transform pre-aligned the clouds so ICP only cleaned up
        // small residuals. Without that prior, ICP must absorb the full ~12 deg inter-view
        // rotation, so the fit quality itself (overlap, coverage, median, rmse) is trusted
        // instead of the correction magnitude. correctionAngleDeg/Translation are still
        // computed and logged as diagnostics.
        const bool trustRefinement = after.valid() && after.overlap >= minOverlapFrac &&
                                     after.spatialCoverage >= 0.5 && improvesMedian && stableRmse &&
                                     wellConstrained;
        const PointCloud &toAppend = trustRefinement ? refined : clouds[i];

        if (trustRefinement)
            std::cout << "  ICP accepted: median " << before.medianDistance << " -> "
                      << after.medianDistance << ", overlap=" << (100.0 * after.overlap)
                      << "%, coverage=" << (100.0 * after.spatialCoverage)
                      << "%, correction=" << correctionAngleDeg << " deg/"
                      << correctionTranslation << " normalized units\n";
        else
            std::cerr << "  ICP rejected; keeping calibration pose. before/after median="
                      << before.medianDistance << "/" << after.medianDistance
                      << ", rmse=" << before.rmse << "/" << after.rmse
                      << ", overlap=" << (100.0 * after.overlap)
                      << "%, coverage=" << (100.0 * after.spatialCoverage)
                      << "%, correction=" << correctionAngleDeg << " deg/"
                      << correctionTranslation << " normalized units.\n";

        // Save this pair's contribution exactly as it enters the fused cloud.
        IcpUtils::saveIndividualCloud(toAppend, cloudPairs[i].first, cloudPairs[i].second, mean0, scale0);

        const auto update = fusion.integrate(toAppend);
        fused = fusion.pointCloud();
        std::cout << "  Fusion update: inserted=" << update.inserted
                  << ", merged=" << update.merged << ", rejected=" << update.rejected
                  << "; model=" << fused.pts.size() << " surfels.\n";
    }

    PlyUtils::denormalise(fused, mean0, scale0);
    PlyUtils::savePLY("pointcloud_fused.ply", fused);

    std::cout << "\nFusion complete. Saved to pointcloud_fused.ply\n";

    // Surface reconstruction: screened Poisson indicator function -> marching cubes.
    // The fused cloud is back in the metric (mm) world frame and already carries the
    // oriented per-point normals accumulated above, so Poisson consumes them directly
    // (no PCA normal estimation). Raise Config::resolution for a finer mesh at the cost
    // of a larger linear solve.
    std::cout << "\n=== Surface reconstruction (Poisson + marching cubes) ===\n";
    PoissonReconstruction poisson;
    IndicatorField field = poisson.computeIndicator(fused);
    if (!field.values.empty())
    {
        Mesh mesh = poisson.extractMesh(field);
        MeshUtils::saveMeshPLY("mesh_fused.ply", mesh);
        std::cout << "Surface reconstruction complete. Saved to mesh_fused.ply\n";
    }
    else
    {
        std::cerr << "Poisson reconstruction produced no indicator field; skipping mesh.\n";
    }

    return 0;
}
