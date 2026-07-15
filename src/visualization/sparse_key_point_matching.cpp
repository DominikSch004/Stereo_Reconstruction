#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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

    std::cout << "Keypoints — left: " << result.keypointsLeft.size()
              << ", right: " << result.keypointsRight.size() << "\n"
              << "Matches passed ratio test: " << result.matches.size() << "\n";

    cv::Mat vis;
    cv::drawMatches(grayLeft, result.keypointsLeft,
                    grayRight, result.keypointsRight,
                    result.matches, vis,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const std::string outPath = "sparse_matches.png";
    cv::imwrite(outPath, vis);
    std::cout << "Saved visualization to " << outPath
              << " (" << result.matches.size() << " of " << result.matches.size()
              << " matches drawn)\n";

    cv::imshow("Sparse Key Point Matching correspondences", vis);
    cv::waitKey(0);
    return 0;
}