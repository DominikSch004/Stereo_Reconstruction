#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "DTULoader.hpp"
#include "SiftFlannMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "ImgUtils.hpp"

int main()
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

    std::vector<cv::Point2f> ptsL, ptsR;
    SiftFlannMatcher::extractPoints(result, ptsL, ptsR);

    std::cout << "Correspondences: " << ptsL.size() << "\n";
    if ((int)ptsL.size() < 8) {
        std::cerr << "Not enough correspondences\n";
        return -1;
    }

    std::vector<bool> inliers;
    Eigen::Matrix3d F = FundamentalMatrix::ransac(ptsL, ptsR, inliers);

    int nIn = std::count(inliers.begin(), inliers.end(), true);
    std::cout << "Inliers: " << nIn << " / " << ptsL.size() << "\n";
    std::cout << "F:\n" << F << "\n";

    // Visualize epipolar lines for first 20 inliers
    cv::Mat vizL, vizR;
    cv::cvtColor(grayLeft,  vizL, cv::COLOR_GRAY2BGR);
    cv::cvtColor(grayRight, vizR, cv::COLOR_GRAY2BGR);

    int drawn = 0;
    for (size_t i = 0; i < ptsL.size() && drawn < 20; ++i)
    {
        if (!inliers[i]) continue;
        cv::Scalar color(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55);
        Eigen::Vector3d p(ptsL[i].x, ptsL[i].y, 1.0);
        Eigen::Vector3d line = F * p;
        int w = grayRight.cols, h = grayRight.rows;
        float y0 = float(-line(2) / line(1));
        float y1 = float(-(line(2) + line(0) * w) / line(1));
        y0 = std::clamp(y0, 0.f, float(h));
        y1 = std::clamp(y1, 0.f, float(h));
        cv::line(vizR, {0, int(y0)}, {w, int(y1)}, color, 1);
        cv::circle(vizL, ptsL[i], 4, color, -1);
        cv::circle(vizR, ptsR[i], 4, color, -1);
        ++drawn;
    }

    cv::Mat combined;
    cv::hconcat(vizL, vizR, combined);
    cv::imshow("Epipolar lines (RANSAC + 8-point)", combined);
    cv::waitKey(0);
    return 0;
}