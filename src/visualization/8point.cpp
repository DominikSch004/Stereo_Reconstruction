#include <iostream>
#include <opencv2/highgui.hpp>
#include <random> // Ensure this is included
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "ImgUtils.hpp"
#include "GeometryUtils.hpp"
#include "Evaluator.hpp"
#include "VisualizationUtils.hpp"
#include "PipelineConfig.hpp"

struct MethodStats
{
    double rot_err_sum = 0.0;
    double trans_err_sum = 0.0;
    double epi_err_sum = 0.0;
    double inlier_ratio_sum = 0.0;
    int valid_count = 0;
};

int main(int argc, char **argv)
{
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

    DTULoader loader("../data/dtu/");

    MethodStats ransacStats, openCVStats, magsacStats, prosacStats;

    const int NUM_PAIRS = 10;
    const int NUM_TRIALS = 50;
    int startLeftImageId = config.imageLeftId;
    int datasetId = config.datasetId;

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int i = startLeftImageId; i <= NUM_PAIRS + startLeftImageId; i++)
    {
        std::cout << "\n--- Processing Pair " << i << " and " << i + 1 << " (" << NUM_TRIALS << " random trials) ---\n";

        StereoPair pair = loader.loadPair(i, i + 1);
        cv::Mat K = loader.loadIntrinsicCV(i);
        CameraPose pose1 = loader.loadCameraPose(i);
        CameraPose pose2 = loader.loadCameraPose(i + 1);

        Eigen::Matrix3d R_gt;
        Eigen::Vector3d t_gt;
        DTULoader::getRelativePose(pose1, pose2, R_gt, t_gt);

        cv::Mat grayLeft = toGray(pair.imageLeft);
        cv::Mat grayRight = toGray(pair.imageRight);

        SparseKeyPointMatcher matcher(0.75f, config.featureDetector);
        MatchResult result = matcher.match(grayLeft, grayRight);

        std::vector<cv::Point2f> ptsL, ptsR;
        SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);

        if (ptsL.size() < 8)
        {
            std::cout << "Not enough points. Skipping pair.\n";
            continue;
        }

        // Run the random trials
        for (int trial = 0; trial < NUM_TRIALS; ++trial)
        {
            VisualizationData visualize;
            std::vector<bool> CustomInliers, OpenCVInliers, Custommagsacinliers, Customprosacinliers;

            Eigen::Matrix3d F_custom = FundamentalMatrix::computeFundamental(ptsL, ptsR, CustomInliers, rng, visualize, FundamentalMethod::CustomRANSAC, 1.0, 0.99, 1000);
            Eigen::Matrix3d F_opencv = FundamentalMatrix::computeFundamental(ptsL, ptsR, OpenCVInliers, rng, visualize, FundamentalMethod::OpenCVRANSAC, 1.0, 0.99, 1000);
            Eigen::Matrix3d F_magsac = FundamentalMatrix::computeFundamental(ptsL, ptsR, Custommagsacinliers, rng, visualize, FundamentalMethod::CustomMAGSAC, 10.0, 0.99, 1000);
            Eigen::Matrix3d F_prosac = FundamentalMatrix::computeFundamental(ptsL, ptsR, Customprosacinliers, rng, visualize, FundamentalMethod::CustomPROSAC, 1.0, 0.99, 1000);

            cv::Mat R_est, t_est;
            double rot_err, trans_err;

                if (GeometryUtils::extractPoseFromFundamental(F_opencv, ptsL, ptsR, OpenCVInliers, K, R_est, t_est))
            {
                Evaluator::evaluatePose(toEigenMat(R_est), toEigenVec(t_est), R_gt, t_gt, rot_err, trans_err);
                openCVStats.rot_err_sum += rot_err;
                openCVStats.trans_err_sum += trans_err;
                openCVStats.epi_err_sum += Evaluator::evaluateEpipolarError(toCvMat(F_opencv), ptsL, ptsR, OpenCVInliers);
                openCVStats.inlier_ratio_sum += Evaluator::computeInlierRatio(OpenCVInliers);
                openCVStats.valid_count++;
            }

            if (GeometryUtils::extractPoseFromFundamental(F_custom, ptsL, ptsR, CustomInliers, K, R_est, t_est))
            {
                Evaluator::evaluatePose(toEigenMat(R_est), toEigenVec(t_est), R_gt, t_gt, rot_err, trans_err);
                ransacStats.rot_err_sum += rot_err;
                ransacStats.trans_err_sum += trans_err;
                ransacStats.epi_err_sum += Evaluator::evaluateEpipolarError(toCvMat(F_custom), ptsL, ptsR, CustomInliers);
                ransacStats.inlier_ratio_sum += Evaluator::computeInlierRatio(CustomInliers);
                ransacStats.valid_count++;
            }

            if (GeometryUtils::extractPoseFromFundamental(F_magsac, ptsL, ptsR, Custommagsacinliers, K, R_est, t_est))
            {
                Evaluator::evaluatePose(toEigenMat(R_est), toEigenVec(t_est), R_gt, t_gt, rot_err, trans_err);
                magsacStats.rot_err_sum += rot_err;
                magsacStats.trans_err_sum += trans_err;
                magsacStats.epi_err_sum += Evaluator::evaluateEpipolarError(toCvMat(F_magsac), ptsL, ptsR, Custommagsacinliers);
                magsacStats.inlier_ratio_sum += Evaluator::computeInlierRatio(Custommagsacinliers);
                magsacStats.valid_count++;
            }

            if (GeometryUtils::extractPoseFromFundamental(F_prosac, ptsL, ptsR, Customprosacinliers, K, R_est, t_est))
            {
                Evaluator::evaluatePose(toEigenMat(R_est), toEigenVec(t_est), R_gt, t_gt, rot_err, trans_err);
                prosacStats.rot_err_sum += rot_err;
                prosacStats.trans_err_sum += trans_err;
                prosacStats.epi_err_sum += Evaluator::evaluateEpipolarError(toCvMat(F_prosac), ptsL, ptsR, Customprosacinliers);
                prosacStats.inlier_ratio_sum += Evaluator::computeInlierRatio(Customprosacinliers);
                prosacStats.valid_count++;
            }
        }
        std::cout << "  Finished " << NUM_TRIALS << " random trials for pair " << i << ".\n";
    }

    // 2. Print Summary Results
    std::cout << "\n***OVERALL AVERAGES***\n";

    auto printAverage = [](const std::string &name, const MethodStats &stats)
    {
        if (stats.valid_count > 0)
        {
            std::cout << "  Average Rotation Error:    " << (stats.rot_err_sum / stats.valid_count) << " deg\n";
            std::cout << "  Average Translation Error: " << (stats.trans_err_sum / stats.valid_count) << " deg\n";
            std::cout << "  Average Epipolar Error:    " << (stats.epi_err_sum / stats.valid_count) << " px\n";
            std::cout << "  Average Inlier Ratio:      " << (stats.inlier_ratio_sum / stats.valid_count) << "%\n\n";
        }
        else
        {
            std::cout << name << " failed to recover pose on all pairs.\n\n";
        }
    };

    printAverage("OpenCV RANSAC", openCVStats);
    printAverage("Custom RANSAC", ransacStats);
    printAverage("MAGSAC", magsacStats);
    printAverage("PROSAC", prosacStats);

    return 0;
}