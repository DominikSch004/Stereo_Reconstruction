#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cstdio>
#include <string>
#include "DTULoader.hpp"
#include "Pipeline.hpp"
#include "PipelineConfig.hpp"
#include "PlyUtils.hpp"
#include "ICP.hpp"

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

    const int numPairs = 3;
    const size_t icpSamples = 4000;

    std::mt19937 rng(42);

    std::vector<PointCloud> clouds;
    std::vector<CameraPose> poses; // one DTU calibration pose per view used as first guess
    clouds.reserve(numPairs);
    poses.reserve(numPairs);
    DTULoader loader("../data/dtu/");

    for (int i = 1; i <= numPairs; ++i)
    {
        std::cout << "\n=== Processing Pair (" << i << ", " << i + 1 << ") ===\n";

        StereoPair pair = loader.loadPair(i, i + 1);
        cv::Mat K = loader.loadIntrinsicCV(i);
        CameraPose poseLeft = loader.loadCameraPose(i);
        CameraPose poseRight = loader.loadCameraPose(i + 1);

        PipelineResult res;
        // Pass both views' camera centers so Pipeline rescales t to the true metric baseline
        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, config,
                                    poseLeft.t, poseRight.t))
        {
            std::cerr << "Pipeline failed for pair " << i << "\n";
            continue;
        }

        PointCloud cloud = PlyUtils::buildPointCloud(res.denseDisparity, res.Q, res.P1r, res.P2r, res.camToWorld, res.rectColor, res.minDisp, res.globalConfidence, config.triangulation);

        std::cout << "Cloud " << i << ": " << cloud.pts.size() << " points generated.\n";
        clouds.push_back(std::move(cloud));
        poses.push_back(poseLeft);
    }

    if (clouds.empty())
    {
        std::cerr << "No point clouds were successfully generated. Aborting execution.\n";
        return -1;
    }

    // Single normalization, derived from cloud 0 only, preserving relative spatial relationship between clouds
    auto [mean0, scale0] = PlyUtils::normalise(clouds[0]);
    for (auto& cloud : clouds)
        for (auto& p : cloud.pts)
            p = (p - mean0) / scale0;

    // Fuse sequentially, seeding each alignment from DTU's known relative pose
    PointCloud fused = clouds[0];
    for (size_t i = 1; i < clouds.size(); ++i)
    {
        std::cout << "\nICP aligning cloud " << i + 1 << " to fused reference...\n";

        // DTU-derived initial guess: relative transform from view i's pose to view 0's pose
        Eigen::Matrix3d R_rel = poses[i].R * poses[0].R.transpose();
        Eigen::Vector3d t_rel = poses[i].R * (poses[0].t - poses[i].t);
        Eigen::Matrix4f seedT = Eigen::Matrix4f::Identity();
        seedT.block<3, 3>(0, 0) = R_rel.cast<float>();
        seedT.block<3, 1>(0, 3) = (t_rel.cast<float>()) / scale0;

        for (auto& pt : clouds[i].pts) {
            Eigen::Vector4f p_h(pt.x(), pt.y(), pt.z(), 1.0f);
            pt = (seedT * p_h).head<3>();
        }
        for (auto& n : clouds[i].normals) {
            n = seedT.block<3, 3>(0, 0) * n;
        }

        PointCloud srcSub = PlyUtils::subsample(clouds[i], icpSamples, rng);
        PointCloud tgtSub = PlyUtils::subsample(fused, icpSamples, rng);
        Eigen::Matrix4f coarseT = ICP::align(srcSub, tgtSub, 30, 0.1f, true, config.icpMode);

        for (auto& pt : clouds[i].pts) {
            Eigen::Vector4f p_h(pt.x(), pt.y(), pt.z(), 1.0f);
            pt = (coarseT * p_h).head<3>();
        }
        for (auto& n : clouds[i].normals) {
            n = coarseT.block<3, 3>(0, 0) * n;
        }

        ICP::align(clouds[i], fused, 20, 0.1f, true, config.icpMode);

        fused.pts.insert(fused.pts.end(), clouds[i].pts.begin(), clouds[i].pts.end());
        fused.colors.insert(fused.colors.end(), clouds[i].colors.begin(), clouds[i].colors.end());
        fused.weights.insert(fused.weights.end(), clouds[i].weights.begin(), clouds[i].weights.end());
        fused.normals.insert(fused.normals.end(), clouds[i].normals.begin(), clouds[i].normals.end());
        fused.validNormal.insert(fused.validNormal.end(), clouds[i].validNormal.begin(), clouds[i].validNormal.end());

        std::cout << "Current fused cloud size: " << fused.pts.size() << " points\n";
    }

    PlyUtils::denormalise(fused, mean0, scale0);
    PlyUtils::savePLY("pointcloud_fused.ply", fused);

    std::cout << "\nFusion complete. Saved to pointcloud_fused.ply\n";
    return 0;
}
