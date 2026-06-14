#pragma once
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include "DTULoader.hpp"
#include "FundamentalMatrix.hpp"

struct PipelineResult
{
    cv::Mat rectLeft, rectRight, rectColor;
    cv::Mat H1, H2;
    std::vector<cv::Point2f> inPtsL, inPtsR;
    cv::Size imgSize;
};

inline cv::Mat toCvMat(const Eigen::Matrix3d& M)
{
    cv::Mat out(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out.at<double>(i, j) = M(i, j);
    return out;
}

inline cv::Point2f applyH(const cv::Mat& H, cv::Point2f p)
{
    cv::Mat pt = (cv::Mat_<double>(3,1) << p.x, p.y, 1.0);
    cv::Mat r  = H * pt;
    return { float(r.at<double>(0) / r.at<double>(2)),
             float(r.at<double>(1) / r.at<double>(2)) };
}

inline void savePLY(const std::string& path,
                    const std::vector<cv::Vec3f>& pts,
                    const std::vector<cv::Vec3b>& colors)
{
    std::ofstream f(path);
    if (!f.is_open())
    {
        std::cerr << "ERROR: Failed to open " << path << " for writing\n";
        return;
    }
    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";
    for (size_t i = 0; i < pts.size(); ++i)
        f << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << " "
          << (int)colors[i][2] << " " << (int)colors[i][1] << " " << (int)colors[i][0] << "\n";
    std::cout << "Saved " << pts.size() << " points to " << path << "\n";
}

// Runs SIFT+FLANN+RANSAC+8point+Loop&Zhang rectification
inline bool runPipeline(const std::string& pathLeft, const std::string& pathRight,
                        PipelineResult& res)
{
    DTULoader loader("");
    StereoPair pair = loader.loadPair(pathLeft, pathRight);
    if (!pair.imageLeft.data || !pair.imageRight.data)
    {
        std::cerr << "ERROR: Failed to load images\n";
        return false;
    }

    cv::Mat colorLeft(pair.imageLeft.h, pair.imageLeft.w, CV_8UC4, pair.imageLeft.data);
    cv::Mat bgrLeft;
    cv::cvtColor(colorLeft, bgrLeft, cv::COLOR_RGBA2BGR);

    auto toGray = [](const FreeImageB& fi) {
        cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
        cv::Mat gray;
        cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
        return gray;
    };

    cv::Mat grayLeft  = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);
    res.imgSize = grayLeft.size();

    // SIFT + FLANN
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

    std::cout << "Correspondences: " << ptsL.size() << "\n";
    if ((int)ptsL.size() < 8) { std::cerr << "Not enough correspondences\n"; return false; }

    // RANSAC + 8-point → F
    std::vector<bool> mask;
    Eigen::Matrix3d F = ransacFundamental(ptsL, ptsR, mask);

    for (size_t i = 0; i < ptsL.size(); ++i)
        if (mask[i]) { res.inPtsL.push_back(ptsL[i]); res.inPtsR.push_back(ptsR[i]); }

    std::cout << "Inliers: " << res.inPtsL.size() << "\n";

    // Loop & Zhang rectification
    if (!cv::stereoRectifyUncalibrated(res.inPtsL, res.inPtsR, toCvMat(F), res.imgSize,
                                       res.H1, res.H2))
    {
        std::cerr << "ERROR: stereoRectifyUncalibrated failed\n";
        return false;
    }

    cv::warpPerspective(grayLeft,  res.rectLeft,  res.H1, res.imgSize);
    cv::warpPerspective(grayRight, res.rectRight, res.H2, res.imgSize);
    cv::warpPerspective(bgrLeft,   res.rectColor, res.H1, res.imgSize);
    return true;
}

// Builds Q matrix (f = image width, cx/cy = center, relative baseline)
inline cv::Mat buildQ(const PipelineResult& res)
{
    double f  = res.imgSize.width;
    double cx = res.imgSize.width  / 2.0;
    double cy = res.imgSize.height / 2.0;
    return (cv::Mat_<double>(4,4) <<
         1,  0,  0,  -cx,
         0,  1,  0,  -cy,
         0,  0,  0,    f,
         0,  0, -1.0,  0);
}

// Reprojects disparity to 3D and saves PLY
inline void buildAndSavePLY(const cv::Mat& dispFloat, const PipelineResult& res,
                             int numDisp, const std::string& plyPath)
{
    cv::Mat disp32f;
    if (dispFloat.type() == CV_32F)
        disp32f = dispFloat;
    else if (dispFloat.type() == CV_16S)
        dispFloat.convertTo(disp32f, CV_32F, 1.0 / 16.0);
    else
        dispFloat.convertTo(disp32f, CV_32F);

    cv::Mat Q = buildQ(res);
    cv::Mat points3D;
    cv::reprojectImageTo3D(disp32f, points3D, Q, true);

    std::vector<cv::Vec3f> pts;
    std::vector<cv::Vec3b> colors;
    const float maxZ = 1e4f;

    for (int y = 0; y < points3D.rows; ++y)
        for (int x = 0; x < points3D.cols; ++x)
        {
            if (disp32f.at<float>(y, x) <= 0.f) continue;
            cv::Vec3f p = points3D.at<cv::Vec3f>(y, x);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            if (std::abs(p[2]) > maxZ) continue;
            pts.push_back(p);
            colors.push_back(res.rectColor.at<cv::Vec3b>(y, x));
        }

    std::cout << "Point cloud: " << pts.size() << " points\n";
    savePLY(plyPath, pts, colors);

    // Show disparity map
    cv::Mat dispViz;
    disp32f.convertTo(dispViz, CV_8U, 255.0 / numDisp);
    cv::applyColorMap(dispViz, dispViz, cv::COLORMAP_TURBO);
    cv::imshow("Disparity", dispViz);
    std::cout << "Press any key to exit...\n";
    cv::waitKey(0);
}
