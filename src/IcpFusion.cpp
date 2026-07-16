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
#include "PoissonReconstruction.hpp"
#include "MeshUtils.hpp"

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

    std::vector<std::pair<int, int>> selectedPairs;
    for (int v = 6; v < 19; ++v)
        selectedPairs.emplace_back(v, v + 1);

    const size_t icpSamples = 4000;      // coarse-stage source subsample
    const size_t icpFineSamples = 30000; // fine-stage source subsample (plenty for 6 DOF)
    const size_t minCloudPoints = 1000; // reject degenerate reconstructions (near-empty clouds)
    const float confidenceKeepFrac = 0.10f;
    const double minOverlapFrac = 0.10;

    std::mt19937 rng(42);

    Eigen::Matrix4f worldAnchor = Eigen::Matrix4f::Identity();

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

        // NO GT pre-registration: the cloud stays in its own RECTIFIED
        // left-camera frame (camToWorld is identity in the pipeline) and ICP
        // alone registers it to the fused reference. The first kept cloud
        // defines the fusion frame; remember its world transform as the
        // output anchor (gauge only, see above).
        if (clouds.empty())
            worldAnchor = IcpUtils::rectToWorldTransform(res.R1, poseLeft);

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

    // Normalize all clouds for more numarical stability of cerer solver
    auto [mean0, scale0] = PlyUtils::normalise(clouds[0]);
    for (size_t ci = 1; ci < clouds.size(); ++ci)
        for (auto &p : clouds[ci].pts)
            p = (p - mean0) / scale0;

    PointCloud fused = clouds[0];
    // Pre-ICP concatenation (clouds at their raw camera-frame placements), kept
    // alongside the ICP result so the registration can be judged against the
    // placement it started from.
    PointCloud fusedPreIcp = clouds[0];
    IcpUtils::saveIndividualCloud(clouds[0], cloudPairs[0].first, cloudPairs[0].second, mean0, scale0, "", worldAnchor);
    for (size_t i = 1; i < clouds.size(); ++i)
    {
        std::cout << "\nICP aligning cloud " << i + 1 << " to fused reference...\n";

        // Refine a working copy so clouds[i] keeps its raw camera-frame placement
        // for the pre-ICP reference cloud and the *_preicp diagnostics.
        PointCloud refined = clouds[i];

        CeresICPOptimizer icp;
        icp.setMode(config.icpMode);

        // Coarse: subsampled source against the FULL fused cloud. A subsampled
        // target would impose an NN-spacing error floor (~1.5 mm at 4000 points)
        // that caps the whole fusion. The 0.5 gate covers the full initial
        // misalignment (~one orbit step between consecutive rectified camera
        // frames, ~0.1-0.3 normalized); the optimizer's adaptive 3x-median gate
        // anneals it as alignment improves.
        PointCloud srcSub = PlyUtils::subsample(refined, icpSamples, rng);
        icp.setMatchingMaxDistance(0.5f);
        icp.setNbOfIterations(40);
        icp.useWeights(false);
        Eigen::Matrix4f coarseT = icp.estimatePose(srcSub, fused);
        IcpUtils::applyRigid(refined, coarseT);

        // Fine: denser source subsample, tight gate (basic unweighted ICP).
        PointCloud srcFine = PlyUtils::subsample(refined, icpFineSamples, rng);
        icp.setMatchingMaxDistance(0.1f);
        icp.setNbOfIterations(30);
        icp.useWeights(true);
        Eigen::Matrix4f fineT = icp.estimatePose(srcFine, fused);
        IcpUtils::applyRigid(refined, fineT);
        int matched = icp.lastMatchCount();

        const double overlapFrac = srcFine.pts.empty()
                                 ? 0.0 : (double)matched / (double)srcFine.pts.size();
        const bool trustRefinement = overlapFrac >= minOverlapFrac;

        // Track the pre-ICP state regardless of acceptance, and save the raw
        // placement for before/after diagnostics.
        IcpUtils::saveIndividualCloud(clouds[i], cloudPairs[i].first, cloudPairs[i].second, mean0, scale0, "_preicp", worldAnchor);
        fusedPreIcp.pts.insert(fusedPreIcp.pts.end(), clouds[i].pts.begin(), clouds[i].pts.end());
        fusedPreIcp.colors.insert(fusedPreIcp.colors.end(), clouds[i].colors.begin(), clouds[i].colors.end());
        fusedPreIcp.weights.insert(fusedPreIcp.weights.end(), clouds[i].weights.begin(), clouds[i].weights.end());
        fusedPreIcp.normals.insert(fusedPreIcp.normals.end(), clouds[i].normals.begin(), clouds[i].normals.end());
        fusedPreIcp.validNormal.insert(fusedPreIcp.validNormal.end(), clouds[i].validNormal.begin(), clouds[i].validNormal.end());

        if (!trustRefinement)
        {
            // Without GT pre-registration there is no fallback placement: the
            // raw camera-frame pose is arbitrary in the fusion frame, so a
            // rejected refinement means the cloud must be dropped entirely.
            std::cerr << "  NOTE: low ICP overlap " << (100.0 * overlapFrac) << "% ("
                      << matched << " correspondences) -- rejecting refinement and "
                      << "SKIPPING this cloud (no registration available without GT poses).\n";
            continue;
        }

        std::cout << "  ICP refined: overlap " << (100.0 * overlapFrac) << "% ("
                  << matched << " correspondences).\n";

        // Save this pair's contribution exactly as it enters the fused cloud.
        IcpUtils::saveIndividualCloud(refined, cloudPairs[i].first, cloudPairs[i].second, mean0, scale0, "", worldAnchor);

        fused.pts.insert(fused.pts.end(), refined.pts.begin(), refined.pts.end());
        fused.colors.insert(fused.colors.end(), refined.colors.begin(), refined.colors.end());
        fused.weights.insert(fused.weights.end(), refined.weights.begin(), refined.weights.end());
        fused.normals.insert(fused.normals.end(), refined.normals.begin(), refined.normals.end());
        fused.validNormal.insert(fused.validNormal.end(), refined.validNormal.begin(), refined.validNormal.end());

        std::cout << "Current fused cloud size: " << fused.pts.size() << " points\n";
    }

    // Outputs move into the DTU world frame via the single gauge anchor (for
    // GT-based evaluation and visualization only; see worldAnchor above).
    PlyUtils::denormalise(fusedPreIcp, mean0, scale0);
    IcpUtils::applyRigid(fusedPreIcp, worldAnchor);
    PlyUtils::savePLY("pointcloud_fused_preicp.ply", fusedPreIcp);

    PlyUtils::denormalise(fused, mean0, scale0);
    IcpUtils::applyRigid(fused, worldAnchor);
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
