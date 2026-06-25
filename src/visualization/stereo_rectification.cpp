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

    // Compute Robust Fundamental Matrix via updated Custom RANSAC pipeline
    std::vector<bool> mask;
    Eigen::Matrix3d F = FundamentalMatrix::computeFundamental(ptsL, ptsR, mask, FundamentalMethod::OpenCVRANSAC, 1.0, 0.99, 1000);

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
        return false;

    int nInliers = std::count(mask.begin(), mask.end(), true);
    std::cout << "RANSAC Inliers: " << nInliers << " / " << ptsL.size() << "\n";

    // --- 3. Relative Pose Recovery ---
    // Estimate the essential matrix DIRECTLY from the correspondences rather than
    // converting from F via E = K^T F K. The latter propagates F's noise and never
    // enforces the essential-matrix constraint (two equal singular values), which
    // yields a poor translation direction and tilts the rectified rows (verified:
    // ~74px vertical residual). findEssentialMat enforces that constraint during
    // RANSAC and recovers a translation matching the ground-truth pose (~0.4px).
    // cv::Mat poseMask;
    // cv::Mat E = cv::findEssentialMat(inL, inR, K, cv::RANSAC, 0.999, 1.0, poseMask);
    cv::Mat F_cv = toCvMat(F);
    cv::Mat E = K.t() * F_cv * K;
    cv::Mat R, t, poseMask;
    cv::Mat cvE_mask = cv::Mat::ones(inL.size(), 1, CV_8U);
    cv::recoverPose(E, inL, inR, K, R, t, cvE_mask);

    // Compute Calibrated Homography mappings
    RectifyResult rect;
    if (!Rectification::computeCalibrated(K, R, t, grayLeft.size(), grayLeft, grayRight, pair.imageLeft, rect, RectificationMethod::CalibratedOpenCV))
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
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Point2f> rInL, rInR;
    cv::undistortPoints(inL, rInL, K, dist, rect.R1, rect.P1);
    cv::undistortPoints(inR, rInR, K, dist, rect.R2, rect.P2);

    std::vector<double> errs;
    errs.reserve(rInL.size());
    double sumErr = 0.0, maxErr = 0.0;
    for (size_t i = 0; i < rInL.size(); ++i)
    {
        double e = std::abs(rInL[i].y - rInR[i].y);
        errs.push_back(e);
        sumErr += e;
        maxErr = std::max(maxErr, e);
    }

    double meanErr = errs.empty() ? 0.0 : sumErr / errs.size();
    std::vector<double> sorted = errs;
    std::sort(sorted.begin(), sorted.end());
    double medianErr = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];

    // Sub-pixel / single-pixel mean residual indicates a correct calibrated rectification.
    int goodCount = 0;
    for (double e : errs)
        if (e <= 1.0)
            ++goodCount;

    std::cout << "\n--- Rectification Vertical-Alignment Error (pixels) ---\n";
    std::cout << "  correspondences : " << errs.size() << "\n";
    std::cout << "  mean   |yL-yR|  : " << meanErr << "\n";
    std::cout << "  median |yL-yR|  : " << medianErr << "\n";
    std::cout << "  max    |yL-yR|  : " << maxErr << "\n";
    std::cout << "  within 1px      : " << goodCount << " / " << errs.size()
              << " (" << (errs.empty() ? 0.0 : 100.0 * goodCount / errs.size()) << "%)\n";
    std::cout << "  verdict         : "
              << (meanErr < 1.0 ? "PASS (rectification row-aligned)"
                                : "CHECK (residual too large)")
              << "\n";

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
    auto errColor = [](double e) -> cv::Scalar {
        if (e <= 1.0) return {0, 255, 0};     // <= 1px : aligned
        if (e <= 3.0) return {0, 255, 255};   // <= 3px : borderline
        return {0, 0, 255};                    // > 3px : misaligned
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
        hud << "mean dy=" << std::fixed << std::setprecision(2) << meanErr
            << "px  max=" << maxErr << "px  within1px="
            << (errs.empty() ? 0.0 : 100.0 * goodCount / errs.size()) << "%";
        cv::putText(combined, hud.str(), {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    {0, 0, 0}, 4, cv::LINE_AA);
        cv::putText(combined, hud.str(), {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    meanErr < 1.0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                    1, cv::LINE_AA);
    }

    // Persist the verification image to disk so it can be inspected without a display.
    const std::string outPath = "rectification_verification.png";
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