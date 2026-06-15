#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "DTULoader.hpp"
#include "FundamentalMatrix.hpp"
#include "MatchSerializer.hpp"
#include "ImgUtils.hpp"
#include "Rectification.hpp"

int main(int argc, char** argv)
{
    std::string leftPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png";
    std::string rightPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png";

    DTULoader loader("");
    StereoPair pair = loader.loadPair(leftPath, rightPath);

    if (!pair.imageLeft.data || !pair.imageRight.data)
    {
        std::cerr << "ERROR: Failed to load images\n";
        return -1;
    }

    cv::Mat grayLeft  = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    std::vector<cv::Point2f> ptsL, ptsR;
    if (!deserializeMatchPoints("matches.bin", ptsL, ptsR)) {
        std::cerr << "ERROR: Failed to deserialize matches. Run sift_flann first.\n";
        return -1;
    }

    std::cout << "Loaded correspondences: " << ptsL.size() << "\n";

    if ((int)ptsL.size() < 8)
    {
        std::cerr << "Not enough correspondences\n";
        return -1;
    }

    // RANSAC + 8-point → F
    std::vector<bool> mask;
    Eigen::Matrix3d F = FundamentalMatrix::ransac(ptsL, ptsR, mask);

    int nInliers = std::count(mask.begin(), mask.end(), true);
    std::cout << "Inliers: " << nInliers << " / " << ptsL.size() << "\n";

    std::vector<cv::Point2f> inL, inR;
    for (size_t i = 0; i < ptsL.size(); ++i)
        if (mask[i]) { inL.push_back(ptsL[i]); inR.push_back(ptsR[i]);
    }

    // Loop & Zhang rectification (stereoRectifyUncalibrated implements Loop & Zhang 1999)
    cv::Mat H1, H2;
    if (!Rectification::computeUncalibrated(inL, inR, grayLeft.size(), Rectification::toCvMat(F), H1, H2))
    {
        std::cerr << "Rectification failed\n";
        return -1;
    }

    std::cout << "H1:\n" << H1 << "\nH2:\n" << H2 << "\n";

    // Warp images
    cv::Mat rectL, rectR;
    Rectification::warp(grayLeft, grayRight, H1, H2, rectL, rectR);

    // Draw horizontal scan lines to verify row-alignment
    cv::Mat vizL, vizR;
    cv::cvtColor(rectL, vizL, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectR, vizR, cv::COLOR_GRAY2BGR);

    for (int y = 0; y < vizL.rows; y += 40)
    {
        cv::line(vizL, {0, y}, {vizL.cols, y}, {0, 255, 0});
        cv::line(vizR, {0, y}, {vizR.cols, y}, {0, 255, 0});
    }

    cv::Mat combined;
    cv::hconcat(vizL, vizR, combined);

    cv::imshow("Rectification", combined);
    cv::waitKey(0);

    return 0;
}
