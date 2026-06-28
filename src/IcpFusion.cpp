#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cstdio>
#include <string>
#include "DTULoader.hpp"
#include "Pipeline.hpp"
#include "PlyUtils.hpp"
#include "ICP.hpp"

int main()
{
    const int scanId = 1;
    const int numPairs = 5;
    const size_t icpSamples = 4000;

    std::mt19937 rng(42);

    std::vector<PointCloud> clouds;
    clouds.reserve(numPairs);
    DTULoader loader("../data/dtu/");

    for (int i = 1; i <= numPairs; ++i)
    {
        std::cout << "\n=== Processing Pair (" << i << ", " << i + 1 << ") ===\n";

        // select by image id, default is dataset 1 (scan1) & illumination 3
        StereoPair pair = loader.loadPair(i, i + 1);
        // get intrinsics of 1st image.
        cv::Mat K = loader.loadIntrinsicCV(i);

        PipelineResult res;
        // Standardize onto updated static scope pipeline execution wrappers
        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, PipelineMode::OpenCV))
        {
            std::cerr << "Pipeline failed for pair " << i << "\n";
            continue;
        }

        // Extract cloud fields using your pipeline's underlying dense tracking layers
        PointCloud cloud = PlyUtils::buildPointCloud(
            res.denseDisparity, res.Q, res.P1r, res.P2r, res.camToWorld, res.rectColor, res.minDisp, res.globalConfidence, TriangulationMethod::OpenCV);

        std::cout << "Cloud " << i << ": " << cloud.pts.size() << " points generated.\n";
        clouds.push_back(std::move(cloud));
    }

    if (clouds.empty())
    {
        std::cerr << "No point clouds were successfully generated. Aborting execution.\n";
        return -1;
    }

    // Normalization shifts mapped seamlessly onto PlyUtils helpers
    auto [mean0, scale0] = PlyUtils::normalise(clouds[0]);
    for (size_t i = 1; i < clouds.size(); ++i)
    {
        PlyUtils::normalise(clouds[i]);
    }

    PointCloud fused = clouds[0];
    for (size_t i = 1; i < clouds.size(); ++i)
    {
        std::cout << "\nICP aligning cloud " << i + 1 << " to fused reference...\n";

        PointCloud srcSub = PlyUtils::subsample(clouds[i], icpSamples, rng);
        PointCloud tgtSub = PlyUtils::subsample(fused, icpSamples, rng);
        Eigen::Matrix4f coarseT = ICP::align(srcSub, tgtSub, 30, 0.1f, true);

        for (auto& pt : clouds[i].pts) {
            Eigen::Vector4f p_h(pt.x(), pt.y(), pt.z(), 1.0f);
            pt = (coarseT * p_h).head<3>();
        }

        ICP::align(clouds[i], fused, 20, 0.1f, true);

        fused.pts.insert(fused.pts.end(), clouds[i].pts.begin(), clouds[i].pts.end());
        fused.colors.insert(fused.colors.end(), clouds[i].colors.begin(), clouds[i].colors.end());
        fused.weights.insert(fused.weights.end(), clouds[i].weights.begin(), clouds[i].weights.end());

        std::cout << "Current fused cloud size: " << fused.pts.size() << " points\n";
    }

    PlyUtils::denormalise(fused, mean0, scale0);
    PlyUtils::savePLY("pointcloud_fused.ply", fused);

    std::cout << "\nFusion complete. Saved to pointcloud_fused.ply\n";
    return 0;
}