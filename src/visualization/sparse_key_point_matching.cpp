#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "ImgUtils.hpp"
#include "PipelineConfig.hpp"
#include "VisualizationUtils.hpp"

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

    int leftImage = config.imageLeftId;
    int rightImage = config.imageRightId;
    int dataset = config.datasetId;

    DTULoader loader("../data/dtu/");

    // select by image id, default is illumination 3
    StereoPair pair = loader.loadPair(leftImage, rightImage, dataset);

    cv::Mat grayLeft = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    SparseKeyPointMatcher matcher(config.ratioThreshold, config.featureDetector);
    MatchResult result = matcher.match(grayLeft, grayRight);

    VisualizationUtils::visualizeSparseKeypoint(result, grayLeft, grayRight);

    return 0;
}