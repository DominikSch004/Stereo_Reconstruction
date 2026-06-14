#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "DTULoader.hpp"

int main(int argc, char** argv)
{
    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    DTULoader loader("");
    StereoPair pair = loader.loadPair(leftPath, rightPath);

    if (!pair.imageLeft.data || !pair.imageRight.data)
    {
        std::cerr << "ERROR: Failed to load images\n";
        return -1;
    }

    auto toGray = [](const FreeImageB& fi) {
        cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
        cv::Mat gray;
        cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
        return gray;
    };

    cv::Mat grayLeft  = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kpLeft, kpRight;
    cv::Mat descLeft, descRight;
    sift->detectAndCompute(grayLeft,  cv::noArray(), kpLeft,  descLeft);
    sift->detectAndCompute(grayRight, cv::noArray(), kpRight, descRight);

    std::cout << "Keypoints — left: " << kpLeft.size() << ", right: " << kpRight.size() << "\n";

    cv::FlannBasedMatcher flann;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    flann.knnMatch(descLeft, descRight, knnMatches, 2);

    const float ratioThresh = 0.75f;
    std::vector<cv::DMatch> goodMatches;
    for (const auto& m : knnMatches)
        if (m[0].distance < ratioThresh * m[1].distance)
            goodMatches.push_back(m[0]);

    std::cout << "Correspondences: " << goodMatches.size() << " / " << knnMatches.size() << " passed ratio test\n";

    for (size_t i = 0; i < goodMatches.size(); ++i)
    {
        const auto& m = goodMatches[i];
        std::cout << "  [" << i << "] left(" << kpLeft[m.queryIdx].pt.x << ", " << kpLeft[m.queryIdx].pt.y
                  << ") -> right(" << kpRight[m.trainIdx].pt.x << ", " << kpRight[m.trainIdx].pt.y
                  << ")  dist=" << m.distance << "\n";
    }

    cv::Mat imgMatches;
    cv::drawMatches(grayLeft, kpLeft, grayRight, kpRight, goodMatches, imgMatches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::namedWindow("SIFT+FLANN Correspondences", cv::WINDOW_AUTOSIZE);
    cv::imshow("SIFT+FLANN Correspondences", imgMatches);
    std::cout << "Press any key to exit...\n";
    cv::waitKey(0);

    return 0;
}
