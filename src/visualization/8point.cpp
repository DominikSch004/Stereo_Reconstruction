#include <iostream>
#include <opencv2/highgui.hpp>
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

    // 1. Load data
    DTULoader loader("../data/dtu/");

    // Initialize accumulators for each active method
    MethodStats ransacStats, openCVStats, magsacStats, prosacStats;

    for (int i = 1; i < 11; i++)
    {
        std::cout << "\n--- Processing Pair " << i << " and " << i + 1 << " ---\n";

        // select by image id, default is dataset 1 (scan1) & illumination 3
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
            continue; // Use continue instead of return -1 so we don't abort the entire run
        }

        std::vector<bool> CustomInliers, OpenCVInliers, Custommagsacinliers, Customprosacinliers;

        // set randon numer from hardware
        std::random_device rd;
        std::mt19937 rng(42);

        Eigen::Matrix3d F_custom = FundamentalMatrix::computeFundamental(ptsL, ptsR, CustomInliers, rng, FundamentalMethod::CustomRANSAC, 1.0, 0.99, 1000);
        rng.seed(42);
        Eigen::Matrix3d F_opencv = FundamentalMatrix::computeFundamental(ptsL, ptsR, OpenCVInliers, rng, FundamentalMethod::OpenCVRANSAC, 1.0, 0.99, 1000);
        // set sigmaMax (user-defined threshold) a lot higher since it is meant as a cut-off
        rng.seed(42);
        Eigen::Matrix3d F_magsac = FundamentalMatrix::computeFundamental(ptsL, ptsR, Custommagsacinliers, rng, FundamentalMethod::CustomMAGSAC, 10.0, 0.99, 1000);
        rng.seed(42);
        Eigen::Matrix3d F_prosac = FundamentalMatrix::computeFundamental(ptsL, ptsR, Customprosacinliers, rng, FundamentalMethod::CustomPROSAC, 1.0, 0.99, 1000);

        Eigen::Matrix3d R_est;
        Eigen::Vector3d t_est;

        // ================= evaluate openCV =================
        double epi_err_opencv = Evaluator::evaluateEpipolarError(toCvMat(F_custom), ptsL, ptsR, OpenCVInliers);
        double opencv_ratio = Evaluator::computeInlierRatio(OpenCVInliers);

        if (GeometryUtils::extractPoseFromFundamental(F_opencv, ptsL, ptsR, OpenCVInliers, K, R_est, t_est))
        {
            double rot_err = 0.0, trans_err = 0.0;
            Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
            std::cout << "OpenCV RANSAC Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
            std::cout << "OpenCV RANSAC Epipolar Error: " << epi_err_opencv << " px\n";
            std::cout << "OpenCV RANSAC Inlier Ratio: " << opencv_ratio << "%\n";

            // Accumulate stats
            openCVStats.rot_err_sum += rot_err;
            openCVStats.trans_err_sum += trans_err;
            openCVStats.epi_err_sum += epi_err_opencv;
            openCVStats.inlier_ratio_sum += opencv_ratio;
            openCVStats.valid_count++;
        }
        else
        {
            std::cout << "OpenCV RANSAC 8-Point Failed to recover pose.\n";
        }

        // ================= evaluate custom RANSAC =================
        double epi_err_custom = Evaluator::evaluateEpipolarError(toCvMat(F_custom), ptsL, ptsR, CustomInliers);
        double custom_ratio = Evaluator::computeInlierRatio(CustomInliers);

        if (GeometryUtils::extractPoseFromFundamental(F_custom, ptsL, ptsR, CustomInliers, K, R_est, t_est))
        {
            double rot_err = 0.0, trans_err = 0.0;
            Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
            std::cout << "RANSAC Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
            std::cout << "RANSAC Epipolar Error: " << epi_err_custom << " px\n";
            std::cout << "RANSAC Inlier Ratio: " << custom_ratio << "%\n";

            // Accumulate stats
            ransacStats.rot_err_sum += rot_err;
            ransacStats.trans_err_sum += trans_err;
            ransacStats.epi_err_sum += epi_err_custom;
            ransacStats.inlier_ratio_sum += custom_ratio;
            ransacStats.valid_count++;
        }
        else
        {
            std::cout << "Custom RANSAC 8-Point Failed to recover pose.\n";
        }

        // ================= evaluate MAGSAC =================
        double epi_err_magsac = Evaluator::evaluateEpipolarError(toCvMat(F_magsac), ptsL, ptsR, Custommagsacinliers);
        double magsac_ratio = Evaluator::computeInlierRatio(Custommagsacinliers);

        if (GeometryUtils::extractPoseFromFundamental(F_magsac, ptsL, ptsR, Custommagsacinliers, K, R_est, t_est))
        {
            double rot_err = 0.0, trans_err = 0.0;
            Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
            std::cout << "MAGSAC Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
            std::cout << "MAGSAC Epipolar Error: " << epi_err_magsac << " px\n";
            std::cout << "MAGSAC Inlier Ratio: " << magsac_ratio << "%\n";

            // Accumulate stats
            magsacStats.rot_err_sum += rot_err;
            magsacStats.trans_err_sum += trans_err;
            magsacStats.epi_err_sum += epi_err_magsac;
            magsacStats.inlier_ratio_sum += magsac_ratio;
            magsacStats.valid_count++;
        }
        else
        {
            std::cout << "MAGSAC 8-point failed to recover pose.\n";
        }

        // ================= evaluate PROSAC =================
        double epi_err_prosac = Evaluator::evaluateEpipolarError(toCvMat(F_prosac), ptsL, ptsR, Customprosacinliers);
        double prosac_ratio = Evaluator::computeInlierRatio(Customprosacinliers);

        if (GeometryUtils::extractPoseFromFundamental(F_prosac, ptsL, ptsR, Customprosacinliers, K, R_est, t_est))
        {
            double rot_err = 0.0, trans_err = 0.0;
            Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
            std::cout << "PROSAC Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
            std::cout << "PROSAC Epipolar Error: " << epi_err_prosac << " px\n";
            std::cout << "PROSAC Inlier Ratio: " << prosac_ratio << "%\n";

            // Accumulate stats
            prosacStats.rot_err_sum += rot_err;
            prosacStats.trans_err_sum += trans_err;
            prosacStats.epi_err_sum += epi_err_prosac;
            prosacStats.inlier_ratio_sum += prosac_ratio;
            prosacStats.valid_count++;
        }
        else
        {
            std::cout << "PROSAC 8-point failed to recover pose.\n";
        }
    }

    // 2. Print Summary Results
    std::cout << "\n================= OVERALL AVERAGES =================\n";

    auto printAverage = [](const std::string &name, const MethodStats &stats)
    {
        if (stats.valid_count > 0)
        {
            std::cout << name << " (Computed over " << stats.valid_count << " successful pairs):\n";
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

    // visualize epipolar matches
    // VisualizationUtils::displayEpipolarMatches("Custom 8-Point", grayLeft, grayRight, ptsL, ptsR, CustomInliers, F_custom);
    // VisualizationUtils::displayEpipolarMatches("OpenCV Baseline", grayLeft, grayRight, ptsL, ptsR, OpenCVInliers, F_opencv);
    // VisualizationUtils::displayEpipolarMatches("MAGSAC", grayLeft, grayRight, ptsL, ptsR, Custommagsacinliers, F_magsac);
    // VisualizationUtils::displayEpipolarMatches("PROSAC", grayLeft, grayRight, ptsL, ptsR, Customprosacinliers, F_prosac);
    // std::cout << "\nExecution complete! Press any key on the image windows to exit.\n";
    // cv::waitKey(0);

    return 0;
}
