#include <iostream>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "Rectification.hpp"
#include "ImgUtils.hpp"
#include "PipelineConfig.hpp"
#include "VisualizationUtils.hpp"
#include "Evaluator.hpp"
#include "GeometryUtils.hpp"

int main(int argc, char **argv)
{
    std::cout << "Initializing Pipeline Evaluation\n";

    std::string out_dir = "../images";
    if (!std::filesystem::exists(out_dir))
    {
        std::filesystem::create_directories(out_dir);
    }

    const std::string configPath = (argc > 1) ? argv[1] : "../config.yaml";
    PipelineConfig config;
    try
    {
        config = PipelineConfig::load(configPath);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
    config.print();

    // Load dataset path
    std::cout << "\nLoading DTU dataset...\n";
    DTULoader loader("../data/dtu/");

    // Load image pair
    StereoPair pair = loader.loadPair(config.imageLeftId, config.imageRightId, config.datasetId, config.illuminationId);
    cv::Mat grayLeft = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    // Load full absolute poses for both cameras
    CameraPose poseLeft = loader.loadCameraPose(config.imageLeftId);
    CameraPose poseRight = loader.loadCameraPose(config.imageRightId);
    cv::Mat K = toCvMat(poseLeft.K);

    // Calculate Ground Truth relative pose
    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    DTULoader::getRelativePose(poseLeft, poseRight, R_gt, t_gt);

    std::cout << "\nInitializing Sparse Feature Matching...\n";

    // Sparse key point matching
    SparseKeyPointMatcher matcher(config.ratioThreshold);
    MatchResult result = matcher.match(grayLeft, grayRight);

    VisualizationUtils::visualizeSparseKeypoint(result, grayLeft, grayRight, out_dir + "/sparseKeypointVisualization.png");

    // 8-point algorithm
    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);
    if (ptsL.size() < 8)
    {
        return -1;
    }

    std::vector<bool> inliers;
    VisualizationData visualization;

    double threshold = 1.0;

    if (config.fundamental == FundamentalMethod::CustomMAGSAC)
    {
        // sigmaMax value not actually threshold
        threshold = 10.0;
    };

    Eigen::Matrix3d F = FundamentalMatrix::computeFundamental(ptsL, ptsR, inliers, config.rng, visualization, config.fundamental, threshold, 0.99, 100000);

    VisualizationUtils::fundamentalExplorationVideo(grayLeft, grayRight, visualization, "Random Sampling Optimization Process", 0, out_dir + "/fundamental_iterations.mp4");

    VisualizationUtils::fundamentalComparison(F, R_gt, t_gt, K);

    cv::Mat F_cv = toCvMat(F);
    cv::Mat E = K.t() * F_cv * K;
    cv::Mat R, t;
    std::vector<cv::Point2f> inL, inR;
    for (size_t i = 0; i < ptsL.size(); ++i)
    {
        if (inliers[i])
        {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
        }
    }
    if (inL.size() < 8)
        return 0;

    cv::recoverPose(E, inL, inR, K, R, t);
    // Evaluation
    EightPointParams initParams = {ptsL, ptsR, F_cv, inliers, R_gt, t_gt, toEigenMat(R), toEigenVec(t)};
    EightPointRes metricsRes = Evaluator::evaluateEightPoint(initParams);
    Evaluator::printEightPoint(metricsRes);

    VisualizationUtils::displayEpipolarMatches("Sample of 20 Epipolar Matches (Before Rectification)", pair.imageLeft, pair.imageRight,
                                               ptsL, ptsR, inliers, F, metricsRes.rot_error_deg, metricsRes.trans_error_deg, metricsRes.epipolar_error,
                                               20, out_dir + "/epipolarMatchesVisualization.png");

    // Optional: non-linear refinement of (R, t) directly on the
    // essential-matrix space
    // See comments in FundamentalMatrix.cpp for details on why we do this
    if (config.refinePose) // flag here is just for comparison, remove it later
    {
        GeometryUtils::refinePose(K, inL, inR, R, t);
        // Recompute E = [t]x * R from the new pose as its used for evaluation
        cv::Mat tx = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2), t.at<double>(1),
                      t.at<double>(2), 0, -t.at<double>(0),
                      -t.at<double>(1), t.at<double>(0), 0);
        E = tx * R;
    }

    cv::Size sz = grayLeft.size();
    cv::Mat bgr1 = pair.imageLeft;

    // --- 4. Stereo Rectification ---
    RectifyResult rect;
    if (!Rectification::computeCalibrated(K, R, t, sz, grayLeft, grayRight, bgr1, rect, config.rectification))
    {
        std::cerr << "ERROR: Stereo rectification execution failure.\n";
        return 0;
    }

    VisualizationUtils::visualizeRectification(rect, inL, inR, K, "Rectification Verification", out_dir + "/rectificationVisualization.png");

    int minDisp = 0;
    int numDisp = 0;

    // Pass the inliers and the individual matrices from the rect struct
    Disparity::computeDynamicSearchRangeCalibrated(
        inL, inR, K,
        rect.R1, rect.P1,
        rect.R2, rect.P2,
        sz, minDisp, numDisp);

    std::cout << "\n--- Disparity Parameters ---\n";
    std::cout << "Dynamic minDisp: " << minDisp << "\n";
    std::cout << "Dynamic numDisp: " << numDisp << "\n";

    // --- 6. Dense Stereo Matching ---
    const int blockSize = 7;

    // Compute the dense disparity map using the rectified images and dynamic bounds
    cv::Mat denseDisparity = Disparity::computeDisparity(
        rect.rectLeft, rect.rectRight,
        minDisp, numDisp,
        blockSize, config.disparity);

    if (denseDisparity.empty())
    {
        std::cerr << "ERROR: Dense stereo matching returned an empty disparity map.\n";
        return 0;
    }

    VisualizationUtils::visualizeDisparity(denseDisparity, rect, inL, inR, K, minDisp, numDisp, "Disparity Verification", out_dir + "/disparityVisualization.png");

    return 0;
}