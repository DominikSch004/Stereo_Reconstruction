#include "Pipeline.hpp"
#include "Triangulation.hpp"
#include "Rectification.hpp"
#include "ImgUtils.hpp"
#include "SiftFlannMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

void savePLY(const std::string& path, const std::vector<cv::Vec3f>& pts, const std::vector<cv::Vec3b>& colors) 
{
    std::ofstream f(path);
    if (!f.is_open()) {
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


bool runPipeline(const std::string& pathLeft, const std::string& pathRight, PipelineResult& res)
{
    DTULoader loader("");
    StereoPair pair = loader.loadPair(pathLeft, pathRight);
    if (!pair.imageLeft.data || !pair.imageRight.data) {
        std::cerr << "ERROR: Failed to load images\n";
        return false;
    }

    cv::Mat bgr1  = toBGR(pair.imageLeft);
    cv::Mat gray1 = toGray(pair.imageLeft);
    cv::Mat gray2 = toGray(pair.imageRight);

    cv::Mat P1 = loader.loadDTUProjection(pathLeft);
    if (P1.empty()) return false;

    cv::Mat K = loader.getIntrinsicFromProjection(P1);

    const double scale = 0.5;
    cv::Size sz(cvRound(gray1.cols * scale), cvRound(gray1.rows * scale));
    cv::resize(gray1, gray1, sz); 
    cv::resize(gray2, gray2, sz);
    cv::resize(bgr1,  bgr1,  sz);
    res.imgSize = sz;

    K.at<double>(0, 0) *= scale; 
    K.at<double>(1, 1) *= scale; 
    K.at<double>(0, 2) *= scale; 
    K.at<double>(1, 2) *= scale; 

    // Extract SIFT features and match with FLANN
    SiftFlannMatcher matcher(0.75f);
    MatchResult matchRes = matcher.match(gray1, gray2);
    
    std::vector<cv::Point2f> ptsL, ptsR;
    SiftFlannMatcher::extractPoints(matchRes, ptsL, ptsR);

    if (ptsL.size() < 8) {
        std::cerr << "ERROR: Insufficient image matching correspondences (" << ptsL.size() << ").\n";
        return false;
    }

    //std::vector<bool> RansacMask;
    //Eigen::Matrix3d F_eigen = FundamentalMatrix::ransac(ptsL, ptsR, RansacMask);

    // Use the opencv implementation of RANSAC + 8-point to get the fundamental matrix and inlier mask
    std::vector<uchar> RansacMask;
    cv::Mat F_cv = cv::findFundamentalMat(ptsL, ptsR, cv::FM_RANSAC, 1.0, 0.99, RansacMask);
    //Eigen::Matrix3d F_eigen;
    //cv::cv2eigen(F_cv, F_eigen);

    // Filter down to active fundamental geometry inliers
    std::vector<cv::Point2f> inL, inR;
    for (size_t i = 0; i < ptsL.size(); ++i) {
        if (RansacMask[i]) {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
        }
    }

    if (inL.size() < 8) {
        std::cerr << "ERROR: Too few geometric inliers remaining after custom RANSAC step.\n";
        return false;
    }

    // E = K^T * F * K
    //cv::Mat F_cv = toCvMat(F_eigen);
    cv::Mat E = K.t() * F_cv * K;
    
    // Recover relative transformations R and t from your computed Essential Matrix
    cv::Mat R, t, poseMask;
    cv::Mat cvE_mask = cv::Mat::ones(inL.size(), 1, CV_8U); // All points passed fundamental RANSAC
    
    // cv::recoverPose(E, points1, points2, cameraMatrix, R, t, mask)
    cv::recoverPose(E, inL, inR, K, R, t, cvE_mask);
    
    poseMask = cvE_mask; 

    res.inPtsL.clear(); res.inPtsR.clear();
    for (int i = 0; i < poseMask.rows; ++i) {
        if (poseMask.at<uchar>(i)) {
            res.inPtsL.push_back(inL[i]);
            res.inPtsR.push_back(inR[i]);
        }
    }

    // Track frame origin mappings back to left camera reference
    res.camToWorld = cv::Mat::zeros(3, 4, CV_64F);
    
    cv::Mat identity3x3 = cv::Mat::eye(3, 3, CV_64F);
    identity3x3.copyTo(res.camToWorld(cv::Rect(0, 0, 3, 3)));

    RectifyResult rect;
    if (!Rectification::computeCalibrated(K, R, t, sz, gray1, gray2, bgr1, rect)) {
        std::cerr << "ERROR: stereo rectification failed.\n";
        return false;
    }
    res.Q         = rect.Q;
    res.P1r       = rect.P1;
    res.P2r       = rect.P2;
    res.rectLeft  = rect.rectLeft;
    res.rectRight = rect.rectRight;
    res.rectColor = rect.rectColor;

    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<float> disps;
    std::vector<cv::Point2f> rL, rR;
    cv::undistortPoints(res.inPtsL, rL, K, dist, rect.R1, rect.P1);
    cv::undistortPoints(res.inPtsR, rR, K, dist, rect.R2, rect.P2);
    for (size_t i = 0; i < rL.size(); ++i) {
        disps.push_back(rL[i].x - rR[i].x);
    }

    if (disps.empty()) return false;
    std::sort(disps.begin(), disps.end());
    float dLo = disps[(size_t)(0.02 * disps.size())];
    float dHi = disps[(size_t)(0.98 * (disps.size() - 1))];
    
    const int margin = 32;
    int dMin = (int)std::floor((dLo - margin) / 16.0) * 16;
    int dMax = (int)std::ceil ((dHi + margin) / 16.0) * 16;
    
    res.minDisp = dMin;
    res.numDisp = std::max(16, ((dMax - dMin + 15) / 16) * 16);

    std::cout << "Clean Modular Pipeline Rectified to size: " << sz << "\n"
              << "Inliers processing: " << res.inPtsL.size() << "\n"
              << "SGBM Parameters assigned -> minDisp: " << res.minDisp << ", numDisp: " << res.numDisp << "\n";
    return true;
}

void buildAndSavePLY(const cv::Mat& dispFloat, const PipelineResult& res, int numDisp, const std::string& plyPath,
                     TriangulationMethod method)
{
    cv::Mat disp32f;
    if (dispFloat.type() == CV_32F)
        disp32f = dispFloat;
    else if (dispFloat.type() == CV_16S)
        dispFloat.convertTo(disp32f, CV_32F, 1.0 / 16.0);
    else
        dispFloat.convertTo(disp32f, CV_32F);

    cv::Mat points3D = reprojectDisparityTo3D(disp32f, res, method);
    if (points3D.empty()) return;

    const cv::Matx33d R = res.camToWorld.colRange(0, 3);
    const cv::Vec3d   t(res.camToWorld.at<double>(0, 3),
                        res.camToWorld.at<double>(1, 3),
                        res.camToWorld.at<double>(2, 3));

    std::vector<cv::Vec3f> pts;
    std::vector<cv::Vec3b> colors;
    const float zMax = 9000.0f;

    for (int y = 0; y < points3D.rows; ++y) {
        for (int x = 0; x < points3D.cols; ++x) {
            if (disp32f.at<float>(y, x) <= (float)res.minDisp) continue;
            
            cv::Vec3f p = points3D.at<cv::Vec3f>(y, x);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            if (p[2] <= 0.f || p[2] > zMax) continue;

            cv::Vec3d w = R * cv::Vec3d(p[0], p[1], p[2]) + t;
            pts.push_back(cv::Vec3f(w[0], w[1], w[2]));
            colors.push_back(res.rectColor.at<cv::Vec3b>(y, x));
        }
    }

    std::cout << "Point cloud generation completed. Staging " << pts.size() << " elements.\n";
    savePLY(plyPath, pts, colors);
}