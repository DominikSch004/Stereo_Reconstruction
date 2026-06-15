#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "DTULoader.hpp"
#include "SiftFlannMatcher.hpp"

static cv::Mat toGray(const FreeImageB& fi)
{
    cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    return gray;
}

int main(int argc, char** argv)
{
    const std::string leftPath  = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png";
    const std::string rightPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png";

    DTULoader loader("");
    StereoPair pair = loader.loadPair(leftPath, rightPath);
    if (!pair.imageLeft.data || !pair.imageRight.data) {
        std::cerr << "ERROR: Failed to load images\n";
        return -1;
    }

    cv::Mat grayLeft  = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    SiftFlannMatcher matcher(0.75f);
    MatchResult result = matcher.match(grayLeft, grayRight);

    std::cout << "Keypoints  — left: " << result.keypointsLeft.size()
              << ", right: "           << result.keypointsRight.size() << "\n"
              << "Matches passed ratio test: " << result.matches.size() << "\n";

    cv::Mat vis;
    cv::drawMatches(grayLeft,  result.keypointsLeft,
                    grayRight, result.keypointsRight,
                    result.matches, vis,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("SIFT + FLANN correspondences", vis);
    std::cout << "Press any key to exit...\n";
    cv::waitKey(0);

    return 0;
}
