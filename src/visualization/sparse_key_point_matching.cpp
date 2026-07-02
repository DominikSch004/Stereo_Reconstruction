#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "ImgUtils.hpp"
#include "PipelineConfig.hpp"

int main(int argc, char **argv)
{
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

    // 1. Load data
    DTULoader loader("../data/dtu/");

    // select by image id, default is dataset 1 (scan1) & illumination 3
    StereoPair pair = loader.loadPair(1, 2);

    cv::Mat grayLeft = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    SparseKeyPointMatcher matcher(0.75f, config.featureDetector);
    MatchResult result = matcher.match(grayLeft, grayRight);

    matcher.visualize(result, grayLeft, grayRight);

    return 0;
}