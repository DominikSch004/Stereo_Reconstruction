#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include "DTULoader.hpp"
#include "Evaluator.hpp"
#include "Pipeline.hpp"

#include "SparseKeyPointMatcher.hpp"
#include "Rectification.hpp"
#include "PipelineConfig.hpp"

#include "ImgUtils.hpp"
#include "VisualizationUtils.hpp"
#include "PlyUtils.hpp"
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

    std::cout << "\nLoading DTU dataset...\n";
    DTULoader loader("../data/dtu/");

    StereoPair pair = loader.loadPair(config.imageLeftId, config.imageRightId, config.datasetId, config.illuminationId);

    CameraPose poseLeft = loader.loadCameraPose(config.imageLeftId);
    CameraPose poseRight = loader.loadCameraPose(config.imageRightId);
    cv::Mat K_in = toCvMat(poseLeft.K);

    // Grayscale conversion + config.processingScale downscale
    // same preprocessing as Pipeline::runPipeline applies, so
    // this runs at the same resolution as every other executable
    // driven by the same config.yaml.
    cv::Mat grayLeft, grayRight, bgrLeft, bgrRight, K;
    cv::Size sz;
    Pipeline::preprocessScale(pair.imageLeft, pair.imageRight, K_in, config.processingScale,
                              grayLeft, grayRight, bgrLeft, bgrRight, K, sz);

    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    DTULoader::getRelativePose(poseLeft, poseRight, R_gt, t_gt);

    std::cout << "\nInitializing Sparse Feature Matching...\n";

    SparseKeyPointMatcher matcher(config.ratioThreshold);
    MatchResult result = matcher.match(grayLeft, grayRight);

    VisualizationUtils::visualizeSparseKeypoint(result, grayLeft, grayRight, out_dir + "/sparseKeypointVisualization.png");

    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);
    if (ptsL.size() < 8)
    {
        return -1;
    }

    std::cout << "\n--- Initializing 8-Point ---\n";

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
    // recoverPose's t is unit-norm; rescale to the true DTU metric baseline (mm) so
    // downstream triangulation comes out at real-world scale, same as Pipeline::runPipeline.
    Pipeline::rescaleToTrueBaseline(poseLeft.t, poseRight.t, t);
    // Evaluation
    EightPointParams initParams = {ptsL, ptsR, F_cv, inliers, R_gt, t_gt, toEigenMat(R), toEigenVec(t)};
    EightPointRes metricsRes = Evaluator::evaluateEightPoint(initParams);
    Evaluator::printEightPoint(metricsRes);

    VisualizationUtils::displayEpipolarMatches("Sample of 20 Epipolar Matches (Before Rectification)", bgrLeft, bgrRight,
                                               ptsL, ptsR, inliers, F, metricsRes.rot_error_deg, metricsRes.trans_error_deg, metricsRes.epipolar_error,
                                               20, out_dir + "/epipolarMatchesVisualization.png");

    // Optional: non-linear refinement of (R, t) directly on the
    // essential-matrix space
    // See comments in FundamentalMatrix.cpp for details on why we do this
    if (config.refinePose) // measured to meaningfully improve dense disparity quality; on by default (see config.yaml)
    {
        GeometryUtils::refinePose(K, inL, inR, R, t);
        // Recompute E = [t]x * R from the new pose as its used for evaluation
        cv::Mat tx = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2), t.at<double>(1),
                      t.at<double>(2), 0, -t.at<double>(0),
                      -t.at<double>(1), t.at<double>(0), 0);
        E = tx * R;
    }

    // refinePose can change t's scale/direction; rescale again
    Pipeline::rescaleToTrueBaseline(poseLeft.t, poseRight.t, t);

    std::cout << "\n--- Stereo Rectification ---\n";
    RectifyResult rect;
    if (!Rectification::computeCalibrated(K, R, t, sz, grayLeft, grayRight, bgrLeft, rect, config.rectification))
    {
        std::cerr << "ERROR: Stereo rectification execution failure.\n";
        return 0;
    }

    RectificationRes rectRes = Evaluator::evaluateRectification(inL, inR, K, rect.R1, rect.P1, rect.R2, rect.P2);
    Evaluator::printRectification(rectRes);

    VisualizationUtils::visualizeRectification(rect, inL, inR, K, rectRes, "Rectification Verification", out_dir + "/rectificationVisualization.png");

    std::cout << "\n--- Dense Stereo Matching ---\n";

    int minDisp = 0;
    int numDisp = 0;

    // Pass the inliers and the individual matrices from the rect struct
    Disparity::computeDynamicSearchRangeCalibrated(
        inL, inR, K,
        rect.R1, rect.P1,
        rect.R2, rect.P2,
        sz, minDisp, numDisp);

    // Disparity
    std::cout << "Dynamic minDisp: " << minDisp << "\n";
    std::cout << "Dynamic numDisp: " << numDisp << "\n";

    const int blockSize = Pipeline::scaledBlockSize(7, config.processingScale);

    // Compute the dense disparity map using the rectified images and dynamic bounds
    cv::Mat denseDisparity = Disparity::computeDisparity(
        rect.rectLeft, rect.rectRight,
        minDisp, numDisp,
        blockSize, config.disparity, config.processingScale,
        config.disparityRefinement);

    if (denseDisparity.empty())
    {
        std::cerr << "ERROR: Dense stereo matching returned an empty disparity map.\n";
        return 0;
    }

    DisparityRes dispRes = Evaluator::evaluateDisparity(denseDisparity, rect.rectLeft, rect.rectRight,
                                                        inL, inR, K, rect.R1, rect.P1, rect.R2, rect.P2,
                                                        minDisp, numDisp);
    Evaluator::printDisparity(dispRes, config.processingScale);

    VisualizationUtils::visualizeDisparity(denseDisparity, rect, inL, inR, K, minDisp, numDisp, dispRes, "Disparity Verification", out_dir + "/disparityVisualization.png");

    cv::Mat dense3DPoints = Triangulation::reprojectDisparityTo3D(denseDisparity, rect.Q, rect.P1, rect.P2, minDisp, config.triangulation);
    if (dense3DPoints.empty())
    {
        std::cerr << "ERROR: 3D point cloud generation failed.\n";
        return 0;
    }

    const float maxValidZ = 9000.0f;
    long validPts = 0;
    double zSum = 0.0, zMin = std::numeric_limits<double>::max(), zMax = std::numeric_limits<double>::lowest();
    for (int y = 0; y < dense3DPoints.rows; ++y)
        for (int x = 0; x < dense3DPoints.cols; ++x)
        {
            const cv::Vec3f &p = dense3DPoints.at<cv::Vec3f>(y, x);
            if (!std::isfinite(p[2]) || p[2] <= 0.0f || p[2] > maxValidZ)
                continue;
            ++validPts;
            zSum += p[2];
            zMin = std::min(zMin, static_cast<double>(p[2]));
            zMax = std::max(zMax, static_cast<double>(p[2]));
        }

    std::cout << "\n--- Triangulation sanity check ---\n";
    std::cout << "  valid 3D points : " << validPts << " / " << (dense3DPoints.rows * dense3DPoints.cols) << "\n";
    if (validPts > 0)
        std::cout << "  depth (Z) range : [" << zMin << ", " << zMax << "] mm, mean " << (zSum / validPts) << "\n";
    else
        std::cerr << "  WARNING: no pixel triangulated to a finite, positive depth.\n";

    std::cout << "\n--- Exporting 3D Point Cloud ---\n";
    std::string plyFilename = out_dir + "/pointcloud.ply";
    std::cout << "Saving cloud to: " << plyFilename << "\n";

    // Grab the pre-calculated inlier ratio directly from your evaluation struct
    float globalConfidence = static_cast<float>(metricsRes.inlier_ratio);
    std::cout << "Global Pair Confidence: " << globalConfidence << "\n";

    if (globalConfidence < 0.5f)
    {
        std::cout << "WARNING: low RANSAC confidence (" << (globalConfidence * 100.0f)
                  << "%), resulting point cloud may be noisy.\n";
    }

    cv::Mat colorizedCloud;
    if (rect.rectLeft.channels() == 1)
    {
        cv::cvtColor(rect.rectLeft, colorizedCloud, cv::COLOR_GRAY2BGR);
    }
    else
    {
        colorizedCloud = rect.rectLeft;
    }

    cv::Mat camToWorld = cv::Mat::eye(3, 4, CV_64F);

    if (!PlyUtils::buildAndSavePLY(
            plyFilename,
            denseDisparity,
            rect.Q,
            rect.P1,
            rect.P2,
            camToWorld,
            colorizedCloud,
            minDisp,
            globalConfidence,
            config.triangulation))
    {
        std::cerr << "ERROR: Point cloud export failed.\n";
        return 0;
    }

    std::cout << "End-to-End Execution Completed Successfully.\n";

    return 0;
}