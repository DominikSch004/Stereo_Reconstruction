#include <iostream>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "ImgUtils.hpp"
#include "PipelineConfig.hpp"
#include "VisualizationUtils.hpp"

int main(int argc, char **argv)
{
    std::cout << "Initializing Pipeline Evaluation\n";

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

    // Load image pair
    std::cout << "\nLoading DTU dataset...\n";
    DTULoader loader("../data/dtu/");

    StereoPair pair = loader.loadPair(config.imageLeftId, config.imageRightId, config.datasetId, config.illuminationId);

    cv::Mat grayLeft = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    std::cout << "\nInitializing Sparse Feature Matching...\n";

    // Sparse key point matching
    SparseKeyPointMatcher matcher(config.ratioThreshold);
    MatchResult result = matcher.match(grayLeft, grayRight);

    matcher.visualize(result, grayLeft, grayRight);

    // 8-point algorithm
    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);
    if (ptsL.size() < 8)
    {
        return -1;
    }

    std::vector<bool> inliers;
    VisualizationData visualization;

    Eigen::Matrix3d F = FundamentalMatrix::computeFundamental(ptsL, ptsR, inliers, visualization, config.fundamental, 1.0, 0.99, 1000);

    VisualizationUtils::fundamentalExplorationVideo(grayLeft, grayRight, visualization, "U-SAC Method", 1);

        return 0;
}