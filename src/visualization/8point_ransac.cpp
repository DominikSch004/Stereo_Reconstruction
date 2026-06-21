#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
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

    SparseKeyPointMatcher matcher(0.75f);
    MatchResult result = matcher.match(grayLeft, grayRight);

    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);

    std::cout << "Correspondences found: " << ptsL.size() << "\n";
    if (ptsL.size() < 8) {
        std::cerr << "Insufficient point pairs available for epipolar calculations\n";
        return -1;
    }

    std::vector<bool> CustomInliers;
    std::vector<bool> OpenCVInliers;
    
    Eigen::Matrix3d F_custom = FundamentalMatrix::computeFundamental(ptsL, ptsR, CustomInliers, FundamentalMethod::CustomRANSAC, 1.0, 0.99, 1000);
    Eigen::Matrix3d F_opencv = FundamentalMatrix::computeFundamental(ptsL, ptsR, OpenCVInliers, FundamentalMethod::OpenCVRANSAC, 1.0, 0.99, 1000);

    int nInCustom = (int)std::count(CustomInliers.begin(), CustomInliers.end(), true);
    int nInOpenCV = (int)std::count(OpenCVInliers.begin(), OpenCVInliers.end(), true);

    std::cout << "\n=== Epipolar Backend Benchmarking ===\n";
    std::cout << "Custom RANSAC Robust Inliers: " << nInCustom << " / " << ptsL.size() << "\n";
    std::cout << "OpenCV RANSAC Robust Inliers: " << nInOpenCV << " / " << ptsL.size() << "\n";
    
    std::cout << "\nComputed Custom Matrix F:\n" << F_custom << "\n";
    std::cout << "\nComputed OpenCV Matrix F:\n" << F_opencv << "\n";

    // Visualize epipolar lines for both methods for first 20 inliers
    
    // Helper
    struct VisualizationTarget {
        const Eigen::Matrix3d& F;
        const std::vector<bool>& inliers;
        cv::Mat vizL;
        cv::Mat vizR;
        std::string title;
    };

    std::vector<VisualizationTarget> targets = {
        {F_custom, CustomInliers, grayLeft.clone(), grayRight.clone(), "Backend: Custom FundamnetalMatrixRANSAC"},
        {F_opencv, OpenCVInliers, grayLeft.clone(), grayRight.clone(), "Backend: Native OpenCV FundamentalMatrixRANSAC"}
    };

    int w = grayRight.cols;
    int h = grayRight.rows;

    for (auto& target : targets) {
        // Re-seed before backend sweeps to guarantee color consistency for matched points
        srand(1337); 
        int drawn = 0;

        for (size_t i = 0; i < ptsL.size() && drawn < 20; ++i) {
            // Generate the color token immediately so rand() steps forward identically 
            // across both loops, regardless of whether this specific point is an inlier or not.
            cv::Scalar color(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55);

            if (!target.inliers[i]) continue;

            // Compute Epipolar Projection Math
            Eigen::Vector3d p(ptsL[i].x, ptsL[i].y, 1.0);
            Eigen::Vector3d line = target.F * p;

            float y0 = float(-line(2) / line(1));
            float y1 = float(-(line(2) + line(0) * w) / line(1));
            y0 = std::clamp(y0, 0.f, float(h));
            y1 = std::clamp(y1, 0.f, float(h));

            // Render geometry marks onto the target's image buffers
            cv::line(target.vizR, cv::Point(0, int(y0)), cv::Point(w, int(y1)), color, 1);
            cv::circle(target.vizL, ptsL[i], 4, color, -1);
            cv::circle(target.vizR, ptsR[i], 4, color, -1);
            ++drawn;
        }
    }

    // Combine channels and pop the final separate displays
    for (const auto& target : targets) {
        cv::Mat combined;
        cv::hconcat(target.vizL, target.vizR, combined);

        cv::namedWindow(target.title, cv::WINDOW_NORMAL);
        cv::imshow(target.title, combined);
    }

    std::cout << "\nUnified loop executed! Windows separated and side-by-side ready.\n";
    std::cout << "Press any key to complete execution.\n";
    cv::waitKey(0);

    return 0;
}