#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include "DTULoader.hpp"
#include "FundamentalMatrix.hpp"

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
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <left_image> <right_image>\n";
        return -1;
    }

    DTULoader loader("");
    StereoPair pair = loader.loadPair(std::string(argv[1]), std::string(argv[2]));

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

    // SIFT + FLANN
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kpLeft, kpRight;
    cv::Mat descLeft, descRight;
    sift->detectAndCompute(grayLeft,  cv::noArray(), kpLeft,  descLeft);
    sift->detectAndCompute(grayRight, cv::noArray(), kpRight, descRight);

    cv::FlannBasedMatcher flann;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    flann.knnMatch(descLeft, descRight, knnMatches, 2);

    const float ratioThresh = 0.75f;
    std::vector<cv::Point2f> ptsL, ptsR;
    for (const auto& m : knnMatches)
        if (m[0].distance < ratioThresh * m[1].distance)
        {
            ptsL.push_back(kpLeft[m[0].queryIdx].pt);
            ptsR.push_back(kpRight[m[0].trainIdx].pt);
        }

    std::cout << "Correspondences after ratio test: " << ptsL.size() << "\n";

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
