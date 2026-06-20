#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cstdio>
#include "DTULoader.hpp"
#include "Pipeline.hpp"
#include "Disparity.hpp"
#include "Cloud.hpp"
#include "ICP.hpp"

std::string getLocalPath(int scanId, int viewId, int illumination = 3)
{
    char viewStr[10];
    snprintf(viewStr, sizeof(viewStr), "%03d", viewId);
    return "../data/dtu/SampleSet/MVS Data/Rectified/scan" + std::to_string(scanId) +
           "/rect_" + std::string(viewStr) + "_" + std::to_string(illumination) + "_r5000.png";
}

int main()
{
    const int scanId    = 1;
    const int numPairs  = 5;
    const int blockSz   = 11;
    const size_t icpSamples = 4000;

    std::mt19937 rng(42);

    std::vector<Cloud> clouds;
    clouds.reserve(numPairs);

    for (int i = 1; i <= numPairs; ++i)
    {
        std::cout << "\n=== Processing Pair (" << i << ", " << i+1 << ") ===\n";
        
        std::string pathLeft  = getLocalPath(scanId, i);
        std::string pathRight = getLocalPath(scanId, i + 1);

        PipelineResult res;
        if (!runPipeline(pathLeft, pathRight, res))
        {
            std::cerr << "Pipeline failed for pair " << i << "\n";
            continue;
        }

        cv::Mat disp = Disparity::computeSAD(res.rectLeft, res.rectRight, res.minDisp, res.numDisp, blockSz);
        
        Cloud cloud = CloudUtils::build(res, disp);
        std::cout << "Cloud " << i << ": " << cloud.pts.size() << " points generated.\n";
        clouds.push_back(std::move(cloud));
    }

    if (clouds.empty()) { std::cerr << "No clouds generated\n"; return -1; }

    auto [mean0, scale0] = CloudUtils::normalise(clouds[0]);
    for (size_t i = 1; i < clouds.size(); ++i) {
        CloudUtils::normalise(clouds[i]);
    }

    Cloud fused = clouds[0];
    for (size_t i = 1; i < clouds.size(); ++i)
    {
        std::cout << "\nICP aligning cloud " << i+1 << " to fused reference...\n";
        
        Cloud srcSub = CloudUtils::subsample(clouds[i], icpSamples, rng);
        Cloud tgtSub = CloudUtils::subsample(fused,     icpSamples, rng);
        ICP::align(srcSub, tgtSub, 30);

        ICP::align(clouds[i], fused, 20);

        fused.pts.insert(fused.pts.end(), clouds[i].pts.begin(), clouds[i].pts.end());
        fused.colors.insert(fused.colors.end(), clouds[i].colors.begin(), clouds[i].colors.end());

        std::cout << "Current fused cloud size: " << fused.pts.size() << " points\n";
    }

    CloudUtils::denormalise(fused, mean0, scale0);

    CloudUtils::savePLY("pointcloud_fused.ply", fused);
    std::cout << "\nFusion complete. Saved to pointcloud_fused.ply\n";
    return 0;
}