#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
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

    // Compute Sparse Matching
    SparseKeyPointMatcher matcher(0.75f);
    MatchResult result = matcher.match(grayLeft, grayRight);
    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);

    std::cout << "Correspondences Found: " << ptsL.size() << "\n";
    if (ptsL.size() < 8) {
        std::cerr << "Not enough correspondences available to continue execution\n";
        return -1;
    }

    // Compute Robust Fundamental Matrix via updated Custom RANSAC pipeline
    std::vector<bool> mask;
    Eigen::Matrix3d F = FundamentalMatrix::computeCustomRANSAC(ptsL, ptsR, mask);

    int nInliers = std::count(mask.begin(), mask.end(), true);
    std::cout << "RANSAC Inliers: " << nInliers << " / " << ptsL.size() << "\n";

    std::vector<cv::Point2f> inL, inR;
    for (size_t i = 0; i < ptsL.size(); ++i) {
        if (mask[i]) { 
            inL.push_back(ptsL[i]); 
            inR.push_back(ptsR[i]); 
        }
    }

    // Compute Uncalibrated Homography mappings
    // TODO: Replace with Calibrated rectification routines after getting intirinsics K
    cv::Mat H1, H2;
    if (!Rectification::computeUncalibrated(inL, inR, grayLeft.size(), toCvMat(F), H1, H2)) {
        std::cerr << "Stereo Rectification calculations failed\n";
        return -1;
    }

    // Warp frames into standard perspective projections
    cv::Mat rectL, rectR;
    Rectification::warp(grayLeft, grayRight, H1, H2, rectL, rectR);

    // Render horizontal baseline verification vectors across tracking rows
    cv::Mat vizL, vizR;
    cv::cvtColor(rectL, vizL, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectR, vizR, cv::COLOR_GRAY2BGR);
    for (int y = 0; y < vizL.rows; y += 40) {
        cv::line(vizL, {0, y}, {vizL.cols, y}, {0, 255, 0}, 1);
        cv::line(vizR, {0, y}, {vizR.cols, y}, {0, 255, 0}, 1);
    }

    cv::Mat combined;
    cv::hconcat(vizL, vizR, combined);
    cv::imshow("Stereo Rectification Row-Alignment Verification", combined);
    cv::waitKey(0);
    return 0;
}