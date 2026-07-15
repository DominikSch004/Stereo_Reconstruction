#include "Pipeline.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "Rectification.hpp"
#include "Disparity.hpp"
#include "ImgUtils.hpp"
#include "GeometryUtils.hpp"
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

void Pipeline::rescaleToTrueBaseline(const Eigen::Vector3d &C1, const Eigen::Vector3d &C2, cv::Mat &t)
{
    double trueBaseline = (C1 - C2).norm();
    if (trueBaseline < 1e-9)
    {
        std::cout << "  [Pipeline] No camera centers provided (baseline = 0) -- keeping unit-norm t, reconstruction is up to scale.\n";
        return;
    }
    std::cout << "  [Pipeline] True DTU baseline: " << trueBaseline << " mm -- rescaling t.\n";

    double tNorm = cv::norm(t);
    if (tNorm > 1e-9)
    {
        t = t * (trueBaseline / tNorm);
    }
    else
    {
        std::cerr << "  [Pipeline] WARNING: recoverPose's t has near-zero norm, cannot rescale.\n";
    }
}

bool Pipeline::runPipeline(const cv::Mat &imgLeft, const cv::Mat &imgRight, const cv::Mat &K_in, PipelineResult &res, const PipelineConfig &config, const Eigen::Vector3d &C1, const Eigen::Vector3d &C2)
{
    // --- 0. Preprocessing: grayscale conversion + half-resolution processing scale ---
    cv::Mat bgr1 = imgLeft.clone();
    cv::Mat gray1, gray2;
    cv::cvtColor(imgLeft, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgRight, gray2, cv::COLOR_BGR2GRAY);

    const double scale = 0.5;
    cv::Size sz(cvRound(gray1.cols * scale), cvRound(gray1.rows * scale));
    cv::resize(gray1, gray1, sz);
    cv::resize(gray2, gray2, sz);
    cv::resize(bgr1, bgr1, sz);
    res.imgSize = sz;

    res.rectColor = bgr1;

    cv::Mat K = K_in.clone();
    K.at<double>(0, 0) *= scale;
    K.at<double>(1, 1) *= scale;
    K.at<double>(0, 2) *= scale;
    K.at<double>(1, 2) *= scale;
    res.K = K.clone();

    // --- 1. Sparse Feature Matching ---
    SparseKeyPointMatcher matcher(0.75f);
    MatchResult matchRes = matcher.match(gray1, gray2);

    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(matchRes, ptsL, ptsR);

    if (ptsL.size() < 8)
    {
        std::cerr << "ERROR: Insufficient feature matches (" << ptsL.size() << ").\n";
        return false;
    }

    // --- 2. Epipolar Geometry & Fundamental Matrix Estimation ---
    std::vector<bool> inlierMask;
    Eigen::Matrix3d F_eigen = FundamentalMatrix::computeFundamental(ptsL, ptsR, inlierMask, config.rng, config.fundamental, 1.0, 0.99, 1000);
    cv::Mat F_cv = toCvMat(F_eigen);
    res.inlierMask = inlierMask;

    if (F_cv.empty())
        return false;

    std::vector<cv::Point2f> inL, inR;
    for (size_t i = 0; i < ptsL.size(); ++i)
    {
        if (inlierMask[i])
        {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
        }
    }
    if (inL.size() < 8)
        return false;

    // --- 3. Relative Pose Recovery ---
    cv::Mat R, t, poseMask;
    cv::Mat E = K.t() * F_cv * K; // E = K^T * F * K
    // No (s, s, 0) SVD projection here: F is already exactly rank-2 by
    // construction (compute8Point/computeWeighted8Point zero the 3rd singular
    // value already). Forcing the two nonzero singular values equal, was
    // already seen to have zero effect on the recovered pose anyway:

    // cv::recoverPose runs its own SVD internally (via decomposeEssentialMat)
    // and only reads U/Vt (for R) and U.col(2) (for t), never D.
    // See opencv/modules/calib3d/src/five-point.cpp.
    // Confirmed via PoseRecoveryComparison (variants B/C, identical either way)

    // The projection was also hurting the Evaluator's Epipolar Error
    // metric: 0.538px with it vs 0.211px without,
    // since we're now using the E that best fits the inlier correspondences,
    // without constraining it unnecessarily. Confirmed via Evaluator.cpp with 
    // pose_refinement=false as refinePose overwrites E.
    
    // TODO: clean up these comments later before final delivery
    cv::recoverPose(E, inL, inR, K, R, t, poseMask);

    res.inPtsL.clear();
    res.inPtsR.clear();
    for (int i = 0; i < poseMask.rows; ++i)
    {
        if (poseMask.at<uchar>(i))
        {
            res.inPtsL.push_back(inL[i]);
            res.inPtsR.push_back(inR[i]);
        }
    }

    // Optional: non-linear refinement of (R, t) directly on the
    // essential-matrix space
    // See comments in FundamentalMatrix.cpp for details on why we do this
    if (config.refinePose) // flag here is just for comparison, remove it later
    {
        GeometryUtils::refinePose(K, res.inPtsL, res.inPtsR, R, t);
        // Recompute E = [t]x * R from the new pose as its used for evaluation
        cv::Mat tx = (cv::Mat_<double>(3, 3) <<
                      0, -t.at<double>(2), t.at<double>(1),
                      t.at<double>(2), 0, -t.at<double>(0),
                      -t.at<double>(1), t.at<double>(0), 0);
        E = tx * R;
    }

    // Rescale t from recoverPose's unit-norm convention to the true DTU metric baseline
    rescaleToTrueBaseline(C1, C2, t);
    // save result for evaluation
    res.R_est = R.clone();
    res.t_est = t.clone();
    res.E = E.clone();

    int inlierCount = static_cast<int>(res.inPtsL.size());
    if (!inL.empty())
        res.globalConfidence = static_cast<float>(inlierCount) / static_cast<float>(inL.size());
    else
        res.globalConfidence = 0.0f;

    std::cout << "Global Pair Confidence: " << res.globalConfidence
              << " (" << inlierCount << "/" << inL.size() << " inliers)\n";
    if (res.globalConfidence < 0.5f)
        std::cout << "WARNING: low RANSAC confidence (" << (res.globalConfidence * 100.0f)
                  << "%), disparity search range may be unreliable.\n";

    res.camToWorld = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat(cv::Mat::eye(3, 3, CV_64F)).copyTo(res.camToWorld(cv::Rect(0, 0, 3, 3)));

    // --- 4. Stereo Rectification ---
    RectifyResult rect;
    if (!Rectification::computeCalibrated(K, R, t, sz, gray1, gray2, bgr1, rect, config.rectification))
    {
        std::cerr << "ERROR: Stereo rectification execution failure.\n";
        return false;
    }
    res.Q = rect.Q;
    res.R1 = rect.R1;
    res.R2 = rect.R2;
    res.P1r = rect.P1;
    res.P2r = rect.P2;
    res.rectLeft = rect.rectLeft;
    res.rectRight = rect.rectRight;
    res.rectColor = rect.rectColor;

    // --- 5. Disparity Bound Dynamic Calculation ---
    // TODO: MAYBE  HAVE SOME METHOD FOR THIS IN DISPARITY/RECTIFICATION CLASSES
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<float> disps;
    std::vector<cv::Point2f> rL, rR;
    cv::undistortPoints(res.inPtsL, rL, K, dist, rect.R1, rect.P1);
    cv::undistortPoints(res.inPtsR, rR, K, dist, rect.R2, rect.P2);
    for (size_t i = 0; i < rL.size(); ++i)
    {
        disps.push_back(rL[i].x - rR[i].x);
    }

    if (disps.empty())
        return false;
    std::sort(disps.begin(), disps.end());
    float dLo = disps[(size_t)(0.02 * disps.size())];
    float dHi = disps[(size_t)(0.98 * (disps.size() - 1))];

    const int margin = 32;
    int dMin = (int)std::floor((dLo - margin) / 16.0) * 16;
    int dMax = (int)std::ceil((dHi + margin) / 16.0) * 16;

    res.minDisp = dMin;
    res.numDisp = std::max(16, ((dMax - dMin + 15) / 16) * 16);

    // Disparity cannot exceed image width; clamp to valid range
    res.minDisp = std::max(res.minDisp, 0);
    res.numDisp = std::min(res.numDisp, sz.width - res.minDisp - 1);
    res.numDisp = std::max(res.numDisp, 16); // minimum meaningful search range
    // OpenCV SGBM requires numDisp divisible by 16
    res.numDisp = (res.numDisp / 16) * 16;

    // --- 6. Dense Stereo Matching ---
    const int blockSize = 7;
    res.denseDisparity = Disparity::computeDisparity(res.rectLeft, res.rectRight, res.minDisp, res.numDisp, blockSize, config.disparity);
    if (res.denseDisparity.empty())
    {
        std::cerr << "ERROR: Dense stereo matching returned an empty disparity map.\n";
        return false;
    }

    // --- 7. Disparity to Depth Reprojection ---
    res.dense3DPoints = Triangulation::reprojectDisparityTo3D(
        res.denseDisparity, res.Q, res.P1r, res.P2r, res.minDisp, config.triangulation);

    std::cout << "[Pipeline] End-to-End Execution Completed Successfully.\n";
    return true;
}
