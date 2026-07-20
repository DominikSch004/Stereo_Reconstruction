#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
#include "ImgUtils.hpp"
#include "GeometryUtils.hpp"
#include "Evaluator.hpp"

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
    Eigen::Matrix3d F = FundamentalMatrix::computeFundamental(ptsL, ptsR, mask, rng, visualize, FundamentalMethod::CustomMAGSAC, 10.0, 0.99, 1000);

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

    // Per-point errors, needed only for coloring the panels below
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Point2f> rInL, rInR;
    cv::undistortPoints(inL, rInL, K, dist, rect.R1, rect.P1);
    cv::undistortPoints(inR, rInR, K, dist, rect.R2, rect.P2);

    std::vector<double> errs;
    errs.reserve(rInL.size());
    for (size_t i = 0; i < rInL.size(); ++i)
        errs.push_back(std::abs(rInL[i].y - rInR[i].y));

    // --- 6. Visualization ---
    cv::Mat vizL, vizR;
    cv::cvtColor(rect.rectLeft, vizL, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rect.rectRight, vizR, cv::COLOR_GRAY2BGR);

    // Faint reference grid so the eye has a horizontal ruler.
    for (int y = 0; y < vizL.rows; y += 40)
    {
        cv::line(vizL, cv::Point(0, y), cv::Point(vizL.cols, y), {60, 60, 60}, 1);
        cv::line(vizR, cv::Point(0, y), cv::Point(vizR.cols, y), {60, 60, 60}, 1);
    }

    // Map a vertical error to a colour: green (good) -> yellow -> red (bad).
    auto errColor = [](double e) -> cv::Scalar
    {
        if (e <= 1.0)
            return {0, 255, 0}; // <= 1px : aligned
        if (e <= 3.0)
            return {0, 255, 255}; // <= 3px : borderline
        return {0, 0, 255};       // > 3px : misaligned
    };

    // Draw the rectified correspondences on each panel.
    for (size_t i = 0; i < rInL.size(); ++i)
    {
        cv::Scalar c = errColor(errs[i]);
        cv::circle(vizL, rInL[i], 4, c, 1, cv::LINE_AA);
        cv::circle(vizR, rInR[i], 4, c, 1, cv::LINE_AA);
    }

    // Combined side-by-side view with connecting lines. A horizontal connector
    // means perfect row alignment; any slope is the vertical error made visible.
    cv::Mat combined;
    cv::hconcat(vizL, vizR, combined);
    const int xOff = vizL.cols;
    for (size_t i = 0; i < rInL.size(); ++i)
    {
        cv::Point pL(cvRound(rInL[i].x), cvRound(rInL[i].y));
        cv::Point pR(cvRound(rInR[i].x) + xOff, cvRound(rInR[i].y));
        cv::line(combined, pL, pR, errColor(errs[i]), 1, cv::LINE_AA);
    }

    // Overlay the summary verdict on the combined image.
    {
        std::ostringstream hud;
        hud << "mean dy=" << std::fixed << std::setprecision(2) << rectRes.meanErr
            << "px  max=" << rectRes.maxErr << "px  within1px="
            << (rectRes.correspondences ? 100.0 * rectRes.within1px / rectRes.correspondences : 0.0) << "%";
        cv::putText(combined, hud.str(), {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    {0, 0, 0}, 4, cv::LINE_AA);
        cv::putText(combined, hud.str(), {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    rectRes.pass ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                    1, cv::LINE_AA);
    }

    // Persist the verification image to disk so it can be inspected without a display.
    const std::string outPath = "rectification_verification_custom.png";
    if (cv::imwrite(outPath, combined))
        std::cout << "\nSaved verification image to: " << outPath << "\n";
    else
        std::cerr << "\nWARNING: failed to write " << outPath << "\n";

    std::cout << "Green lines = aligned (<=1px), yellow = borderline, red = misaligned.\n";

    // Show interactively only when a display is available (skipped on headless runs).
    if (std::getenv("DISPLAY") != nullptr)
    {
        cv::namedWindow("Rectification Verification (Combined)", cv::WINDOW_NORMAL);
        cv::imshow("Rectification Verification (Combined)", combined);
        std::cout << "Press any key in the window to exit.\n";
        cv::waitKey(0);
    }

    return 0;
}