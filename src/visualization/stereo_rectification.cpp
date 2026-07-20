#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
#include "ImgUtils.hpp"
#include "GeometryUtils.hpp"
#include "Evaluator.hpp"
#include "VisualizationUtils.hpp"

int main()
{
    // 1. Load data
    DTULoader loader("../data/dtu/");

    // select by image id, default is dataset 1 (scan1) & illumination 3
    StereoPair pair = loader.loadPair(1, 2);
    cv::Mat K = loader.loadIntrinsicCV(1);

    cv::Mat grayLeft = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    // Compute Sparse Matching
    SparseKeyPointMatcher matcher(0.75f);
    MatchResult result = matcher.match(grayLeft, grayRight);
    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);

    std::cout << "Correspondences Found: " << ptsL.size() << "\n";
    if (ptsL.size() < 8)
    {
        std::cerr << "Not enough correspondences available to continue execution\n";
        return -1;
    }

    VisualizationData visualize;
    // Compute Robust Fundamental Matrix via Custom MAGSAC
    std::vector<bool> mask;

    std::mt19937 rng(42);
    Eigen::Matrix3d F = FundamentalMatrix::computeFundamental(ptsL, ptsR, mask, rng, visualize, FundamentalMethod::CustomMAGSAC, 1.0, 0.99, 1000);

    std::vector<cv::Point2f> inL, inR;
    for (size_t i = 0; i < ptsL.size(); ++i)
    {
        if (mask[i])
        {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
        }
    }
    if (inL.size() < 8)
        return 0;

    int nInliers = std::count(mask.begin(), mask.end(), true);
    std::cout << "MAGSAC Inliers: " << nInliers << " / " << ptsL.size() << "\n";

    // --- 3. Relative Pose Recovery ---
    cv::Mat R, t, poseMask;
    cv::Mat F_cv = toCvMat(F);
    cv::Mat E = K.t() * F_cv * K; // E = K^T * F * K
    // No (s, s, 0) SVD projection: it has zero effect on recoverPose's output
    // See pipeline.cpp for details
    cv::recoverPose(E, inL, inR, K, R, t, poseMask);
    // Non-linear pose refinement
    // (see GeometryUtils.hpp / Pipeline.cpp / FundamentalMatrix.cpp for details)
    GeometryUtils::refinePose(K, inL, inR, R, t);

    // Compute Calibrated Homography mappings
    RectifyResult rect;
    if (!Rectification::computeCalibrated(K, R, t, grayLeft.size(), grayLeft, grayRight, pair.imageLeft, rect, RectificationMethod::CalibratedCustom))
    {
        std::cerr << "Stereo Rectification calculations failed\n";
        return -1;
    }

    // --- 5. Quantitative rectification verification ---
    // Rectification is correct iff a correspondence in the left image lands on the
    // SAME row in the right image. We therefore push the RANSAC inlier matches
    // through the rectifying transforms (R1/P1, R2/P2) and measure the residual
    // vertical disparity |y_L - y_R|. This is the actual verification; the drawn
    // lines below are only a rendering of these numbers.
    RectificationRes rectRes = Evaluator::evaluateRectification(inL, inR, K, rect.R1, rect.P1, rect.R2, rect.P2);
    Evaluator::printRectification(rectRes);

    // --- 6. Visualization ---
    const std::string outPath = "rectification_verification_custom.png";
    VisualizationUtils::visualizeRectification(rect, inL, inR, K, rectRes,
                                               "Rectification Verification", outPath);

    return 0;
}