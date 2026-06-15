#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include "DTULoader.hpp"
#include "PairPipeline.hpp"
#include "Disparity.hpp"
#include "Cloud.hpp"
#include "ICP.hpp"

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main()
{
    const std::string datasetPath = "../data/dtu";
    const int scanId    = 1;
    const int numPairs  = 5;
    const int maxDisp   = 128;
    const int blockSz   = 11;
    const size_t icpSamples = 4000;

    DTULoader loader(datasetPath);
    std::mt19937 rng(42);

    std::vector<Cloud> clouds;
    clouds.reserve(numPairs);

    for (int i = 1; i <= numPairs; ++i)
    {
        std::cout << "\n=== Pair (" << i << ", " << i+1 << ") ===\n";
        StereoPair pair = loader.loadPair(scanId, i, i + 1);

        PipelineResult res;
        if (!runPipeline(pair, res))
        {
            std::cerr << "Pipeline failed for pair " << i << "\n";
            continue;
        }


        // HERE CHANGE THIS TO YOUR DESIRED DISPARITY METHOD (NCC, SAD, SSD)
        std::cout << "SAD disparity...\n";
        cv::Mat disp = Disparity::computeSAD(res.rectLeft, res.rectRight, maxDisp, blockSz);

        //std::cout << "NCC disparity...\n";
        //cv::Mat disp = computeNCC(res.rectLeft, res.rectRight, maxDisp, blockSz);

        //std::cout << "SSD disparity...\n";
        //cv::Mat disp = computeSSD(res.rectLeft, res.rectRight, maxDisp, blockSz); 
        

        Cloud cloud = CloudUtils::build(res, disp);
        std::cout << "Cloud " << i << ": " << cloud.pts.size() << " points\n";
        clouds.push_back(std::move(cloud));
    }

    if (clouds.empty()) { std::cerr << "No clouds generated\n"; return -1; }

    // Normalise all clouds to the same scale for ICP
    std::vector<std::pair<Eigen::Vector3f, float>> norms(clouds.size());
    for (size_t i = 0; i < clouds.size(); ++i)
        norms[i] = CloudUtils::normalise(clouds[i]);

    // ICP: align each cloud to the accumulated fused cloud
    Cloud fused = clouds[0];

    for (size_t i = 1; i < clouds.size(); ++i)
    {
        std::cout << "\nICP aligning cloud " << i+1 << " to fused cloud...\n";
        Cloud srcSub = CloudUtils::subsample(clouds[i], icpSamples, rng);
        Cloud tgtSub = CloudUtils::subsample(fused,     icpSamples, rng);

        ICP::align(srcSub, tgtSub);

        // Apply the same transform found on subsampled source to full cloud
        // by re-running ICP on full source using the subsampled result as warm start
        ICP::align(clouds[i], tgtSub, 10);

        // Append to fused
        for (size_t j = 0; j < clouds[i].pts.size(); ++j)
        {
            fused.pts.push_back(clouds[i].pts[j]);
            fused.colors.push_back(clouds[i].colors[j]);
        }

        std::cout << "Fused cloud size: " << fused.pts.size() << " points\n";
    }

    CloudUtils::savePLY("pointcloud_fused.ply", fused);
    return 0;
}
