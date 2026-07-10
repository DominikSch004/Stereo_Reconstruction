#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <utility>
#include <cmath>
#include <Eigen/Dense>
#include "DTULoader.hpp"
#include "Pipeline.hpp"
#include "PipelineConfig.hpp"
#include "PlyUtils.hpp"
#include "ICP.hpp"
#include "IcpUtils.hpp"

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

    const std::vector<std::pair<int, int>> selectedPairs = {
        {1, 2}, {4, 5}, {7, 8}, {12, 13}, {18, 19}, {26, 27}, {32, 33}, {37, 38}};

    const size_t icpSamples = 4000;
    const size_t minCloudPoints = 1000; // reject degenerate reconstructions (near-empty clouds)
    const float confidenceKeepFrac = 0.10f;
    const double minOverlapFrac = 0.10;

    std::mt19937 rng(42);

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

        PointCloud cloud = PlyUtils::buildPointCloud(res.denseDisparity, res.Q, res.P1r, res.P2r, res.camToWorld, res.rectColor, res.minDisp, res.globalConfidence, config.triangulation);

        const size_t beforeCull = cloud.pts.size();
        const size_t culled = IcpUtils::cullByConfidence(cloud, confidenceKeepFrac);
        if (culled > 0)
            std::cout << "  Confidence cull (keep weight >= " << confidenceKeepFrac
                      << " x max): removed " << culled << " / " << beforeCull << " ("
                      << (100.0 * (double)culled / (double)beforeCull) << "%), "
                      << cloud.pts.size() << " kept.\n";

        if (cloud.pts.size() < minCloudPoints)
        {
            std::cerr << "WARNING: pair (" << leftView << "," << rightView << ") produced only "
                      << cloud.pts.size() << " points (< " << minCloudPoints
                      << ") -- degenerate reconstruction, skipping.\n";
            continue;
        }

        // The cloud lives in the pair's RECTIFIED left-camera frame (camToWorld is identity in
        // the pipeline). Place it in the shared DTU world frame so the clouds are co-registered
        // by calibration and ICP only has to correct residual pipeline error.
        IcpUtils::transformCloudToWorld(cloud, res.R1, poseLeft);

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

    PointCloud fused = clouds[0];
    IcpUtils::saveIndividualCloud(clouds[0], cloudPairs[0].first, cloudPairs[0].second, mean0, scale0);
    for (size_t i = 1; i < clouds.size(); ++i)
    {
        std::cout << "\nICP aligning cloud " << i + 1 << " to fused reference...\n";

        // Refine a working copy so clouds[i] retains its untouched calibration placement as fallback.
        PointCloud refined = clouds[i];

        PointCloud srcSub = PlyUtils::subsample(refined, icpSamples, rng);
        PointCloud tgtSub = PlyUtils::subsample(fused, icpSamples, rng);

        CeresICPOptimizer icp;
        icp.setMode(config.icpMode);
        icp.setMatchingMaxDistance(0.1f);

        // Coarse alignment on subsampled clouds (unweighted).
        icp.setNbOfIterations(40);
        icp.useWeights(false);
        Eigen::Matrix4f coarseT = icp.estimatePose(srcSub, tgtSub);
        IcpUtils::applyRigid(refined, coarseT);

        // Fine refinement against the full fused cloud (confidence-weighted).
        icp.setNbOfIterations(20);
        icp.useWeights(true);
        Eigen::Matrix4f fineT = icp.estimatePose(refined, fused);
        IcpUtils::applyRigid(refined, fineT);
        int matched = icp.lastMatchCount();

        const double overlapFrac = clouds[i].pts.empty()
                                 ? 0.0 : (double)matched / (double)clouds[i].pts.size();
        const bool trustRefinement = overlapFrac >= minOverlapFrac;
        const PointCloud &toAppend = trustRefinement ? refined : clouds[i];

        if (trustRefinement)
            std::cout << "  ICP refined: overlap " << (100.0 * overlapFrac) << "% ("
                      << matched << " correspondences).\n";
        else
            std::cerr << "  NOTE: low ICP overlap " << (100.0 * overlapFrac) << "% ("
                      << matched << " correspondences) -- appending at calibration placement "
                      << "without refinement.\n";

        // Save this pair's contribution exactly as it enters the fused cloud.
        IcpUtils::saveIndividualCloud(toAppend, cloudPairs[i].first, cloudPairs[i].second, mean0, scale0);

        fused.pts.insert(fused.pts.end(), toAppend.pts.begin(), toAppend.pts.end());
        fused.colors.insert(fused.colors.end(), toAppend.colors.begin(), toAppend.colors.end());
        fused.weights.insert(fused.weights.end(), toAppend.weights.begin(), toAppend.weights.end());
        fused.normals.insert(fused.normals.end(), toAppend.normals.begin(), toAppend.normals.end());
        fused.validNormal.insert(fused.validNormal.end(), toAppend.validNormal.begin(), toAppend.validNormal.end());

        std::cout << "Current fused cloud size: " << fused.pts.size() << " points\n";
    }

    PlyUtils::denormalise(fused, mean0, scale0);
    PlyUtils::savePLY("pointcloud_fused.ply", fused);

    std::cout << "\nFusion complete. Saved to pointcloud_fused.ply\n";
    return 0;
}
