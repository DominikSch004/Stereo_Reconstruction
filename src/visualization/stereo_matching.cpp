#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
#include "Disparity.hpp"
#include "ImgUtils.hpp"

// Utility function to normalize and colorize disparity maps for visualization
static cv::Mat cleanViz(const cv::Mat& disp) {
    cv::Mat viz;
    cv::normalize(disp, viz, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(viz, viz, cv::COLORMAP_JET);
    return viz;
}

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

    // Convert raw inputs to grayscale for matching matrices
    cv::Mat grayL = toGray(pair.imageLeft);
    cv::Mat grayR = toGray(pair.imageRight);

    // Compute Sparse Feature Alignment
    SparseKeyPointMatcher matcher(0.75f);
    MatchResult result = matcher.match(grayL, grayR);
    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);

    std::cout << "Correspondences Found: " << ptsL.size() << "\n";
    if (ptsL.size() < 8) {
        std::cerr << "ERROR: Not enough correspondences available to proceed.\n";
        return -1;
    }

    // Compute Robust Fundamental Matrix via updated Custom RANSAC pipeline
    std::vector<bool> mask;
    Eigen::Matrix3d F = FundamentalMatrix::computeCustomRANSAC(ptsL, ptsR, mask);

    // Isolate clean inliers to ensure clean rectification homographies
    std::vector<cv::Point2f> inL, inR;
    for (size_t i = 0; i < ptsL.size(); ++i) {
        if (mask[i]) { 
            inL.push_back(ptsL[i]); 
            inR.push_back(ptsR[i]); 
        }
    }

    // Image Plane Rectification
    // Compute Uncalibrated Homography mappings
    // TODO: Replace with Calibrated rectification routines after getting intirinsics K
    cv::Mat H1, H2;
    if (!Rectification::computeUncalibrated(inL, inR, grayL.size(), toCvMat(F), H1, H2)) {
        std::cerr << "ERROR: Stereo Rectification matrix processing failed.\n";
        return -1;
    }

    // Warp raw frames into a guaranteed horizontal row-aligned perspective
    cv::Mat rectL, rectR;
    Rectification::warp(grayL, grayR, H1, H2, rectL, rectR);

    // Hardcode dataset boundaries (DTU setup parameters)
    int minDisp = 40;
    int numDisp = 64; 
    int blockSize = 7;

    // Define the testing suite parameters using our unified DisparityMethod enum pairs
    std::vector<std::pair<DisparityMethod, std::string>> methods = {
        {DisparityMethod::SAD,        "Custom SAD"},
        {DisparityMethod::SSD,        "Custom SSD"},
        {DisparityMethod::NCC,        "Custom NCC"},
        {DisparityMethod::OpenCVSGBM, "OpenCV SGBM Baseline"}
    };

    std::cout << "\n=== Executing Uniform Cost Volume Processing Suite ===\n";

    for (const auto& [method, windowTitle] : methods) {
        std::cout << "Computing cost map for: " << windowTitle << "...\n";
        
        cv::Mat rawDisp = Disparity::computeDisparity(rectL, rectR, minDisp, numDisp, blockSize, method);
        
        // Colorize
        cv::Mat coloredViz = cleanViz(rawDisp);

        // Render to its own scalable independent window window
        cv::namedWindow("Metric: " + windowTitle, cv::WINDOW_NORMAL);
        cv::imshow("Metric: " + windowTitle, coloredViz);
    }

    std::cout << "\nAll windows separated successfully! Drag them around to compare.\n";
    std::cout << "Press any key to close visualization panels and exit.\n";
    cv::waitKey(0);

    return 0;
}
