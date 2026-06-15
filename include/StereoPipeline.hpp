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
    cv::Mat Q;            // 4x4 disparity-to-depth matrix (rectified-left camera frame)
    cv::Mat camToWorld;   // 3x4 [R|t]: rectified-left camera coords -> world (stl) frame
    int minDisp = 0;      // SGBM min disparity derived from the calibrated geometry
    int numDisp = 16;     // SGBM disparity range (multiple of 16)
    std::vector<cv::Point2f> inPtsL, inPtsR;
    cv::Size imgSize;
};

inline cv::Vec3d triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2) {

	// system of equations assuming image=[u,v] and X=[x,y,z,1]
	// from u(p3.X)= p1.X and v(p3.X)=p2.X
	cv::Matx43d A(u1(0)*p1.at<double>(2, 0) - p1.at<double>(0, 0),
		u1(0)*p1.at<double>(2, 1) - p1.at<double>(0, 1),
		u1(0)*p1.at<double>(2, 2) - p1.at<double>(0, 2),
		u1(1)*p1.at<double>(2, 0) - p1.at<double>(1, 0),
		u1(1)*p1.at<double>(2, 1) - p1.at<double>(1, 1),
		u1(1)*p1.at<double>(2, 2) - p1.at<double>(1, 2),
		u2(0)*p2.at<double>(2, 0) - p2.at<double>(0, 0),
		u2(0)*p2.at<double>(2, 1) - p2.at<double>(0, 1),
		u2(0)*p2.at<double>(2, 2) - p2.at<double>(0, 2),
		u2(1)*p2.at<double>(2, 0) - p2.at<double>(1, 0),
		u2(1)*p2.at<double>(2, 1) - p2.at<double>(1, 1),
		u2(1)*p2.at<double>(2, 2) - p2.at<double>(1, 2));

	cv::Matx41d B(p1.at<double>(0, 3) - u1(0)*p1.at<double>(2, 3),
		p1.at<double>(1, 3) - u1(1)*p1.at<double>(2, 3),
		p2.at<double>(0, 3) - u2(0)*p2.at<double>(2, 3),
		p2.at<double>(1, 3) - u2(1)*p2.at<double>(2, 3));

	// X contains the 3D coordinate of the reconstructed point
	cv::Vec3d X;
	// solve AX=B
	cv::solve(A, B, X, cv::DECOMP_SVD);
	return X;
}

// triangulate a vector of image points
inline void triangulate_points(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D) {

	for (int i = 0; i < pts1.size(); i++) {

		pts3D.push_back(triangulate(p1, p2, pts1[i], pts2[i]));
	}
}

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
    cv::Vec3d pt(p.x, p.y, 1.0);
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

// Loads the DTU ground-truth 3x4 camera projection matrix for a rectified image.
// Given ".../MVS Data/Rectified/scanX/rect_001_3_r5000.png" it reads the matching
// ".../MVS Data/Calibration/cal18/pos_001.txt".
inline cv::Mat loadDTUProjection(const std::string& imgPath)
{
    size_t rpos = imgPath.find("Rectified");
    size_t fpos = imgPath.find("rect_");
    if (rpos == std::string::npos || fpos == std::string::npos)
    {
        std::cerr << "ERROR: cannot parse view id from " << imgPath << "\n";
        return cv::Mat();
    }
    std::string base = imgPath.substr(0, rpos);          // ".../MVS Data/"
    std::string id   = imgPath.substr(fpos + 5, 3);      // "001"
    std::string calPath = base + "Calibration/cal18/pos_" + id + ".txt";

    std::ifstream f(calPath);
    if (!f.is_open())
    {
        std::cerr << "ERROR: cannot open calibration file " << calPath << "\n";
        return cv::Mat();
    }
    cv::Mat P(3, 4, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            if (!(f >> P.at<double>(i, j)))
            {
                std::cerr << "ERROR: malformed calibration file " << calPath << "\n";
                return cv::Mat();
            }
    return P;
}

// Calibrated two-view stereo. Uses ONLY the provided intrinsics: matches features,
// estimates the essential matrix, recovers the relative pose, then rectifies and
// fills res with the rectified pair, Q, and a data-driven disparity range.
// The reconstruction is up to an unknown scale (translation is a unit vector).
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

    auto toBGR = [](const FreeImageB& fi) {
        cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data), bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        return bgr;
    };
    auto toGray = [](const FreeImageB& fi) {
        cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data), gray;
        cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
        return gray;
    };

    cv::Mat bgr1  = toBGR(pair.imageLeft),  bgr2  = toBGR(pair.imageRight);
    cv::Mat gray1 = toGray(pair.imageLeft), gray2 = toGray(pair.imageRight);

    // Provided intrinsics: take K from the calibration projection matrix (the
    // intrinsic part only; the dataset extrinsics are deliberately NOT used).
    cv::Mat P1 = loadDTUProjection(pathLeft);
    if (P1.empty()) return false;
    cv::Mat K, Rtmp, ctmp;
    cv::decomposeProjectionMatrix(P1, K, Rtmp, ctmp);
    K.convertTo(K, CV_64F);

    // Work at reduced resolution for speed (intrinsics scale with the image).
    const double scale = 0.5;
    cv::Size sz(cvRound(gray1.cols * scale), cvRound(gray1.rows * scale));
    cv::resize(gray1, gray1, sz); cv::resize(gray2, gray2, sz);
    cv::resize(bgr1,  bgr1,  sz); cv::resize(bgr2,  bgr2,  sz);
    K.rowRange(0, 2) *= scale;

    // SIFT + FLANN + Lowe ratio test.
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kpL, kpR;
    cv::Mat descL, descR;
    sift->detectAndCompute(gray1, cv::noArray(), kpL, descL);
    sift->detectAndCompute(gray2, cv::noArray(), kpR, descR);

    cv::FlannBasedMatcher flann;
    std::vector<std::vector<cv::DMatch>> knn;
    flann.knnMatch(descL, descR, knn, 2);

    std::vector<cv::Point2f> ptsL, ptsR;
    for (const auto& m : knn)
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance)
        {
            ptsL.push_back(kpL[m[0].queryIdx].pt);
            ptsR.push_back(kpR[m[0].trainIdx].pt);
        }
    std::cout << "Matches: " << ptsL.size() << "\n";
    if ((int)ptsL.size() < 8) { std::cerr << "Not enough matches\n"; return false; }

    // Essential matrix from the calibrated correspondences, then relative pose.
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(ptsL, ptsR, K, cv::RANSAC, 0.999, 1.0, mask);
    if (E.empty()) { std::cerr << "ERROR: findEssentialMat failed\n"; return false; }
    cv::Mat R, t;
    int nInl = cv::recoverPose(E, ptsL, ptsR, K, R, t, mask);
    std::cout << "recoverPose inliers: " << nInl << ", t = " << t.t() << "\n";

    // Rectify with the recovered pose. alpha=-1 keeps a sane (near-native) focal.
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat R1r, R2r, P1r, P2r, Q;
    cv::stereoRectify(K, dist, K, dist, sz, R, t,
                      R1r, R2r, P1r, P2r, Q, cv::CALIB_ZERO_DISPARITY, -1);

    cv::Mat mapAx, mapAy, mapBx, mapBy;
    cv::initUndistortRectifyMap(K, dist, R1r, P1r, sz, CV_16SC2, mapAx, mapAy);
    cv::initUndistortRectifyMap(K, dist, R2r, P2r, sz, CV_16SC2, mapBx, mapBy);
    cv::remap(gray1, res.rectLeft,  mapAx, mapAy, cv::INTER_LINEAR);
    cv::remap(gray2, res.rectRight, mapBx, mapBy, cv::INTER_LINEAR);
    cv::remap(bgr1,  res.rectColor, mapAx, mapAy, cv::INTER_LINEAR);

    res.Q = Q;
    res.imgSize = sz;
    res.camToWorld = cv::Mat::eye(3, 4, CV_64F);   // cloud stays in the rectified-left frame

    // Data-driven disparity range: rectify the inlier matches and look at xL - xR.
    std::vector<cv::Point2f> inL, inR;
    for (int i = 0; i < mask.rows; ++i)
        if (mask.at<uchar>(i)) { inL.push_back(ptsL[i]); inR.push_back(ptsR[i]); }
    std::vector<cv::Point2f> rL, rR;
    cv::undistortPoints(inL, rL, K, dist, R1r, P1r);
    cv::undistortPoints(inR, rR, K, dist, R2r, P2r);
    std::vector<float> disps;
    disps.reserve(rL.size());
    for (size_t i = 0; i < rL.size(); ++i) disps.push_back(rL[i].x - rR[i].x);
    std::sort(disps.begin(), disps.end());
    float dLo = disps[(size_t)(0.02 * disps.size())];
    float dHi = disps[(size_t)(0.98 * (disps.size() - 1))];
    const int margin = 32;
    int dMin = (int)std::floor((dLo - margin) / 16.0) * 16;
    int dMax = (int)std::ceil ((dHi + margin) / 16.0) * 16;
    res.minDisp = dMin;
    res.numDisp = std::max(16, ((dMax - dMin + 15) / 16) * 16);

    std::cout << "Rectified " << sz << ", focal " << P1r.at<double>(0, 0)
              << " px, disparity [" << dLo << ", " << dHi << "]\n";
    std::cout << "SGBM minDisp " << res.minDisp << ", numDisp " << res.numDisp << "\n";
    return true;
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

    // Reproject with Q -> 3D in the rectified-left camera frame (up to scale).
    cv::Mat points3D;
    cv::reprojectImageTo3D(disp32f, points3D, res.Q, true);

    // Optional camToWorld transform (identity for the up-to-scale two-view cloud).
    const cv::Matx33d R = res.camToWorld.colRange(0, 3);
    const cv::Vec3d   t(res.camToWorld.at<double>(0, 3),
                        res.camToWorld.at<double>(1, 3),
                        res.camToWorld.at<double>(2, 3));

    std::vector<cv::Vec3f> pts;
    std::vector<cv::Vec3b> colors;
    const float zMax = 9000.0f;   // reprojectImageTo3D marks invalid depth as 10000

    for (int y = 0; y < points3D.rows; ++y)
        for (int x = 0; x < points3D.cols; ++x)
        {
            if (disp32f.at<float>(y, x) <= (float)res.minDisp) continue;
            cv::Vec3f p = points3D.at<cv::Vec3f>(y, x);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            if (p[2] <= 0.f || p[2] > zMax) continue;           // keep points in front
            cv::Vec3d w = R * cv::Vec3d(p[0], p[1], p[2]) + t;
            pts.push_back(cv::Vec3f(w[0], w[1], w[2]));
            colors.push_back(res.rectColor.at<cv::Vec3b>(y, x));
        }

    std::cout << "Point cloud: " << pts.size() << " points\n";
    savePLY(plyPath, pts, colors);

    // Show the disparity map (skip when running headless, e.g. over SSH).
    if (std::getenv("DISPLAY"))
    {
        cv::Mat dispViz;
        disp32f.convertTo(dispViz, CV_8U, 255.0 / numDisp);
        cv::applyColorMap(dispViz, dispViz, cv::COLORMAP_TURBO);
        cv::imshow("Disparity", dispViz);
        std::cout << "Press any key to exit...\n";
        cv::waitKey(0);
    }
}
