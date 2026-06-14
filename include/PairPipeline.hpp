#pragma once
#include "StereoPipeline.hpp"

// Runs the full pipeline from a StereoPair loaded via DTULoader
inline bool runPipeline(const StereoPair& pair, PipelineResult& res)
{
    if (!pair.imageLeft.data || !pair.imageRight.data)
    {
        std::cerr << "ERROR: Failed to load stereo pair\n";
        return false;
    }

    auto toGray = [](const FreeImageB& fi) {
        cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
        cv::Mat gray;
        cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
        return gray;
    };

    cv::Mat grayLeft  = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);
    cv::Mat colorLeft(pair.imageLeft.h, pair.imageLeft.w, CV_8UC4, pair.imageLeft.data);
    cv::Mat bgrLeft;
    cv::cvtColor(colorLeft, bgrLeft, cv::COLOR_RGBA2BGR);

    res.imgSize = grayLeft.size();

    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kpL, kpR;
    cv::Mat descL, descR;
    sift->detectAndCompute(grayLeft,  cv::noArray(), kpL, descL);
    sift->detectAndCompute(grayRight, cv::noArray(), kpR, descR);

    cv::FlannBasedMatcher flann;
    std::vector<std::vector<cv::DMatch>> knn;
    flann.knnMatch(descL, descR, knn, 2);

    std::vector<cv::Point2f> ptsL, ptsR;
    for (const auto& m : knn)
        if (m[0].distance < 0.75f * m[1].distance)
        {
            ptsL.push_back(kpL[m[0].queryIdx].pt);
            ptsR.push_back(kpR[m[0].trainIdx].pt);
        }

    if ((int)ptsL.size() < 8) { std::cerr << "Not enough correspondences\n"; return false; }

    std::vector<bool> mask;
    Eigen::Matrix3d F = ransacFundamental(ptsL, ptsR, mask);

    for (size_t i = 0; i < ptsL.size(); ++i)
        if (mask[i]) { res.inPtsL.push_back(ptsL[i]); res.inPtsR.push_back(ptsR[i]); }

    cv::stereoRectifyUncalibrated(res.inPtsL, res.inPtsR, toCvMat(F), res.imgSize,
                                  res.H1, res.H2);

    cv::warpPerspective(grayLeft,  res.rectLeft,  res.H1, res.imgSize);
    cv::warpPerspective(grayRight, res.rectRight, res.H2, res.imgSize);
    cv::warpPerspective(bgrLeft,   res.rectColor, res.H1, res.imgSize);
    return true;
}
