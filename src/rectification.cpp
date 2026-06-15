#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "DTULoader.hpp"
#include "FundamentalMatrix.hpp"
#include "MatchSerializer.hpp"
#include "ImgUtils.hpp"

// Convert Eigen 3x3 to cv::Mat (CV_64F)
static cv::Mat toCvMat(const Eigen::Matrix3d& M)
{
    cv::Mat out(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out.at<double>(i, j) = M(i, j);
    return out;
}

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
    std::vector<bool> inlierMask;
    Eigen::Matrix3d F = ransacFundamental(ptsL, ptsR, inlierMask);

    int nInliers = std::count(inlierMask.begin(), inlierMask.end(), true);
    std::cout << "Inliers: " << nInliers << " / " << ptsL.size() << "\n";

    std::vector<cv::Point2f> inPtsL, inPtsR;
    for (size_t i = 0; i < ptsL.size(); ++i)
        if (inlierMask[i]) { inPtsL.push_back(ptsL[i]); inPtsR.push_back(ptsR[i]); }

    // Loop & Zhang rectification (stereoRectifyUncalibrated implements Loop & Zhang 1999)
    cv::Mat H1, H2;
    cv::stereoRectifyUncalibrated(inPtsL, inPtsR, toCvMat(F), grayLeft.size(), H1, H2);

    std::cout << "H1:\n" << H1 << "\nH2:\n" << H2 << "\n";

    // Warp images
    cv::Mat rectLeft, rectRight;
    cv::warpPerspective(grayLeft,  rectLeft,  H1, grayLeft.size());
    cv::warpPerspective(grayRight, rectRight, H2, grayRight.size());

    // Draw horizontal scan lines to verify row-alignment
    cv::Mat vizLeft, vizRight;
    cv::cvtColor(rectLeft,  vizLeft,  cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectRight, vizRight, cv::COLOR_GRAY2BGR);
    for (int y = 0; y < vizLeft.rows; y += 40)
    {
        cv::line(vizLeft,  {0, y}, {vizLeft.cols,  y}, cv::Scalar(0, 200, 0), 1);
        cv::line(vizRight, {0, y}, {vizRight.cols, y}, cv::Scalar(0, 200, 0), 1);
    }

    cv::Mat combined;
    cv::hconcat(vizLeft, vizRight, combined);
    cv::namedWindow("Rectified — Loop & Zhang 1999", cv::WINDOW_AUTOSIZE);
    cv::imshow("Rectified — Loop & Zhang 1999", combined);
    std::cout << "Press any key to exit...\n";
    cv::waitKey(0);

    return 0;
}
