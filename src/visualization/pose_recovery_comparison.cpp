// Compares pose-recovery variants against DTU ground truth (#11).
//
// Variants evaluated on the SAME fundamental matrix / inlier set per pair:
//   [A] cv::findEssentialMat + recoverPose        (previous pipeline approach)
//   [B] E = K^T F K, SVD-projected (rank 2, equal sigmas) + recoverPose (current pipeline)
//   [C] E = K^T F K, raw, no projection + recoverPose (isolates the effect of the projection)
//
// Metrics per variant:
//   - rotation error vs DTU ground-truth relative pose (degrees)
//   - translation direction error vs ground truth (degrees)
//   - rectification vertical residual |yL - yR| over the F-inliers (pixels)
//
// Output: metrics table on stdout for all evaluated pairs, plus a stacked
// rectification visualization (one row per variant) for the first pair.

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
#include "ImgUtils.hpp"

namespace
{

struct VariantResult
{
    std::string name;
    cv::Mat R, t;
    double rotErrDeg = -1.0;
    double transErrDeg = -1.0;
    double meanDy = -1.0, medianDy = -1.0, maxDy = -1.0;
    double within1px = 0.0;
    bool rectOk = false;
    cv::Mat panel; // rectified side-by-side visualization row
};

double rotationErrorDeg(const cv::Mat &R_est, const Eigen::Matrix3d &R_gt)
{
    Eigen::Matrix3d Re;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Re(i, j) = R_est.at<double>(i, j);
    Eigen::Matrix3d dR = Re * R_gt.transpose();
    double c = (dR.trace() - 1.0) / 2.0;
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c) * 180.0 / CV_PI;
}

double translationErrorDeg(const cv::Mat &t_est, const Eigen::Vector3d &t_gt)
{
    Eigen::Vector3d te(t_est.at<double>(0), t_est.at<double>(1), t_est.at<double>(2));
    if (te.norm() < 1e-12 || t_gt.norm() < 1e-12)
        return -1.0;
    double c = te.normalized().dot(t_gt.normalized());
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c) * 180.0 / CV_PI;
}

// SVD projection onto the essential-matrix manifold: rank 2, equal singular values.
cv::Mat projectToEssential(const cv::Mat &E_hat)
{
    cv::SVD svd(E_hat);
    double sigma = (svd.w.at<double>(0) + svd.w.at<double>(1)) / 2.0;
    cv::Mat w = cv::Mat::zeros(3, 1, CV_64F);
    w.at<double>(0) = sigma;
    w.at<double>(1) = sigma;
    return svd.u * cv::Mat::diag(w) * svd.vt;
}

cv::Scalar errColor(double e)
{
    if (e <= 1.0) return {0, 255, 0};
    if (e <= 3.0) return {0, 255, 255};
    return {0, 0, 255};
}

} // namespace

int main(int argc, char **argv)
{
    DTULoader loader("../data/dtu/");
    std::vector<std::pair<int, int>> viewPairs = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};
    if (argc == 3)
        viewPairs = {{std::atoi(argv[1]), std::atoi(argv[2])}};

    std::vector<std::vector<VariantResult>> allResults;
    cv::Mat firstPairViz;

    for (size_t p = 0; p < viewPairs.size(); ++p)
    {
        const int idL = viewPairs[p].first;
        const int idR = viewPairs[p].second;
        std::cout << "\n========== Pair (" << idL << ", " << idR << ") ==========\n";

        StereoPair pair = loader.loadPair(idL, idR);
        if (pair.imageLeft.empty() || pair.imageRight.empty())
            return -1;
        cv::Mat K = loader.loadIntrinsicCV(idL);

        // Ground-truth relative pose (left -> right, same convention as recoverPose)
        CameraPose poseL = loader.loadCameraPose(idL);
        CameraPose poseR = loader.loadCameraPose(idR);
        Eigen::Matrix3d R_gt;
        Eigen::Vector3d t_gt;
        DTULoader::getRelativePose(poseL, poseR, R_gt, t_gt);

        cv::Mat grayLeft = toGray(pair.imageLeft);
        cv::Mat grayRight = toGray(pair.imageRight);

        SparseKeyPointMatcher matcher(0.75f);
        MatchResult result = matcher.match(grayLeft, grayRight);
        std::vector<cv::Point2f> ptsL, ptsR;
        SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);
        if (ptsL.size() < 8)
        {
            std::cerr << "Not enough correspondences, skipping pair\n";
            continue;
        }

        // Same F estimator as the pipeline config (custom_magsac); every variant
        // below sees the identical F / inlier set, so differences are purely
        // attributable to the pose-recovery step.
        std::vector<bool> mask;
        Eigen::Matrix3d F = FundamentalMatrix::computeFundamental(
            ptsL, ptsR, mask, FundamentalMethod::CustomMAGSAC, 1.0, 0.99, 1000);
        cv::Mat F_cv = toCvMat(F);

        std::vector<cv::Point2f> inL, inR;
        for (size_t i = 0; i < ptsL.size(); ++i)
        {
            if (mask[i])
            {
                inL.push_back(ptsL[i]);
                inR.push_back(ptsR[i]);
            }
        }
        std::cout << "Correspondences: " << ptsL.size() << ", F-inliers: " << inL.size() << "\n";
        if (inL.size() < 8)
        {
            std::cerr << "Not enough inliers, skipping pair\n";
            continue;
        }

        // --- Build the three essential-matrix variants ---
        std::vector<VariantResult> variants(3);
        {
            variants[0].name = "A: findEssentialMat";
            cv::Mat m;
            cv::Mat E = cv::findEssentialMat(inL, inR, K, cv::RANSAC, 0.999, 1.0, m);
            cv::recoverPose(E, inL, inR, K, variants[0].R, variants[0].t, m);
        }
        {
            variants[1].name = "B: E=K'FK projected";
            cv::Mat E = projectToEssential(K.t() * F_cv * K);
            cv::Mat m;
            cv::recoverPose(E, inL, inR, K, variants[1].R, variants[1].t, m);
        }
        {
            variants[2].name = "C: E=K'FK raw";
            cv::Mat E = K.t() * F_cv * K;
            cv::Mat m;
            cv::recoverPose(E, inL, inR, K, variants[2].R, variants[2].t, m);
        }

        // --- Evaluate each variant: pose error vs GT + rectification residual ---
        std::vector<cv::Mat> panels;
        for (auto &v : variants)
        {
            v.rotErrDeg = rotationErrorDeg(v.R, R_gt);
            v.transErrDeg = translationErrorDeg(v.t, t_gt);

            RectifyResult rect;
            v.rectOk = Rectification::computeCalibrated(
                K, v.R, v.t, grayLeft.size(), grayLeft, grayRight, pair.imageLeft,
                rect, RectificationMethod::CalibratedOpenCV);
            if (!v.rectOk)
            {
                std::cerr << v.name << ": rectification failed\n";
                continue;
            }

            cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
            std::vector<cv::Point2f> rInL, rInR;
            cv::undistortPoints(inL, rInL, K, dist, rect.R1, rect.P1);
            cv::undistortPoints(inR, rInR, K, dist, rect.R2, rect.P2);

            std::vector<double> errs;
            errs.reserve(rInL.size());
            double sum = 0.0;
            v.maxDy = 0.0;
            int good = 0;
            for (size_t i = 0; i < rInL.size(); ++i)
            {
                double e = std::abs(rInL[i].y - rInR[i].y);
                errs.push_back(e);
                sum += e;
                v.maxDy = std::max(v.maxDy, e);
                if (e <= 1.0)
                    ++good;
            }
            v.meanDy = errs.empty() ? 0.0 : sum / errs.size();
            std::vector<double> sorted = errs;
            std::sort(sorted.begin(), sorted.end());
            v.medianDy = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
            v.within1px = errs.empty() ? 0.0 : 100.0 * good / errs.size();

            // Visualization row for the first pair only
            if (p == 0)
            {
                cv::Mat vizL, vizR;
                cv::cvtColor(rect.rectLeft, vizL, cv::COLOR_GRAY2BGR);
                cv::cvtColor(rect.rectRight, vizR, cv::COLOR_GRAY2BGR);
                for (int y = 0; y < vizL.rows; y += 40)
                {
                    cv::line(vizL, {0, y}, {vizL.cols, y}, {60, 60, 60}, 1);
                    cv::line(vizR, {0, y}, {vizR.cols, y}, {60, 60, 60}, 1);
                }
                for (size_t i = 0; i < rInL.size(); ++i)
                {
                    cv::Scalar c = errColor(errs[i]);
                    cv::circle(vizL, rInL[i], 4, c, 1, cv::LINE_AA);
                    cv::circle(vizR, rInR[i], 4, c, 1, cv::LINE_AA);
                }
                cv::Mat row;
                cv::hconcat(vizL, vizR, row);
                const int xOff = vizL.cols;
                for (size_t i = 0; i < rInL.size(); ++i)
                {
                    cv::Point pL(cvRound(rInL[i].x), cvRound(rInL[i].y));
                    cv::Point pR(cvRound(rInR[i].x) + xOff, cvRound(rInR[i].y));
                    cv::line(row, pL, pR, errColor(errs[i]), 1, cv::LINE_AA);
                }
                std::ostringstream hud;
                hud << v.name << "   rotErr=" << std::fixed << std::setprecision(3)
                    << v.rotErrDeg << "deg  tErr=" << v.transErrDeg
                    << "deg  mean|dy|=" << std::setprecision(2) << v.meanDy
                    << "px  within1px=" << std::setprecision(1) << v.within1px << "%";
                cv::putText(row, hud.str(), {15, 45}, cv::FONT_HERSHEY_SIMPLEX, 1.2,
                            {0, 0, 0}, 8, cv::LINE_AA);
                cv::putText(row, hud.str(), {15, 45}, cv::FONT_HERSHEY_SIMPLEX, 1.2,
                            v.meanDy < 1.0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                            2, cv::LINE_AA);
                v.panel = row;
                panels.push_back(row);
            }
        }

        if (p == 0 && !panels.empty())
        {
            cv::vconcat(panels, firstPairViz);
            cv::resize(firstPairViz, firstPairViz, {}, 0.5, 0.5, cv::INTER_AREA);
        }

        // --- Per-pair table ---
        std::cout << "\n" << std::left << std::setw(24) << "variant"
                  << std::right << std::setw(12) << "rotErr[deg]"
                  << std::setw(12) << "tErr[deg]"
                  << std::setw(12) << "mean|dy|"
                  << std::setw(12) << "median|dy|"
                  << std::setw(12) << "max|dy|"
                  << std::setw(12) << "<=1px[%]" << "\n";
        for (const auto &v : variants)
        {
            std::cout << std::left << std::setw(24) << v.name << std::right << std::fixed
                      << std::setprecision(4)
                      << std::setw(12) << v.rotErrDeg
                      << std::setw(12) << v.transErrDeg
                      << std::setprecision(3)
                      << std::setw(12) << v.meanDy
                      << std::setw(12) << v.medianDy
                      << std::setw(12) << v.maxDy
                      << std::setprecision(1) << std::setw(12) << v.within1px << "\n";
        }
        allResults.push_back(variants);
    }

    // --- Aggregate over all pairs ---
    if (allResults.size() > 1)
    {
        std::cout << "\n========== Mean over " << allResults.size() << " pairs ==========\n";
        std::cout << std::left << std::setw(24) << "variant"
                  << std::right << std::setw(12) << "rotErr[deg]"
                  << std::setw(12) << "tErr[deg]"
                  << std::setw(12) << "mean|dy|" << "\n";
        for (size_t vi = 0; vi < 3; ++vi)
        {
            double rot = 0, tr = 0, dy = 0;
            for (const auto &pr : allResults)
            {
                rot += pr[vi].rotErrDeg;
                tr += pr[vi].transErrDeg;
                dy += pr[vi].meanDy;
            }
            const double n = static_cast<double>(allResults.size());
            std::cout << std::left << std::setw(24) << allResults[0][vi].name
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(12) << rot / n
                      << std::setw(12) << tr / n
                      << std::setprecision(3) << std::setw(12) << dy / n << "\n";
        }
    }

    if (!firstPairViz.empty())
    {
        const std::string outPath = "pose_recovery_comparison.png";
        if (cv::imwrite(outPath, firstPairViz))
            std::cout << "\nSaved comparison visualization to: " << outPath << "\n";
        else
            std::cerr << "\nWARNING: failed to write " << outPath << "\n";
    }

    return 0;
}
