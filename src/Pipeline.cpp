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

    // RANSAC + 8-point → fundamental matrix and inlier mask.
    // Switch FundamentalMethod::OpenCV <-> Manual to choose the backend.
    std::vector<uchar> RansacMask;
    cv::Mat F_cv = FundamentalMatrix::compute(ptsL, ptsR, RansacMask,
                                              FundamentalMethod::OpenCV);

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