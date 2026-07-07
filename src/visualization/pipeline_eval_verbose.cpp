#include <iostream>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "ImgUtils.hpp"
#include "PipelineConfig.hpp"
#include "VisualizationUtils.hpp"
#include "Evaluator.hpp"
#include "GeometryUtils.hpp"

int main(int argc, char **argv)
{
    std::cout << "Initializing Pipeline Evaluation\n";

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

    matcher.visualize(result, grayLeft, grayRight);

    // 8-point algorithm
    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);
    if (ptsL.size() < 8)
    {
        return -1;
    }

    std::vector<bool> inliers;
    VisualizationData visualization;

    Eigen::Matrix3d F = FundamentalMatrix::computeFundamental(ptsL, ptsR, inliers, visualization, config.fundamental, 1.0, 0.99, 1000);

    VisualizationUtils::fundamentalExplorationVideo(grayLeft, grayRight, visualization, "U-SAC Method");

    VisualizationUtils::fundamentalComparison(F, R_gt, t_gt, K);

    cv::Mat R, t;
    GeometryUtils::extractPoseFromFundamental(F, ptsL, ptsR, inliers, K, R, t);

    double rot_error_deg, trans_error_deg;
    Eigen::Matrix3d R_est = toEigenMat(R);
    Eigen::Vector3d t_est = toEigenVec(t);
    Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_error_deg, trans_error_deg);

    double epipolar_err = Evaluator::evaluateEpipolarError(F, ptsL, ptsR, inliers);

    VisualizationUtils::displayEpipolarMatches("Epipolar Matches", pair.imageLeft, pair.imageRight, ptsL, ptsR, inliers, F, rot_error_deg, trans_error_deg, epipolar_err);

        return 0;
}