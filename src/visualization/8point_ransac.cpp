#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "ImgUtils.hpp"
#include "GeometryUtils.hpp"
#include "Evaluator.hpp"
#include "VisualizationUtils.hpp"

int main()
{
    // 1. Load data
    DTULoader loader("../data/dtu/");

    // Sweep values: 8, 9, 10, 15, 20, 30, 40, ... 400
    std::vector<int> inlierNumValues = {8, 9, 10, 15};
    for (int v = 20; v <= 400; v += 10)
        inlierNumValues.push_back(v);

    std::ofstream resultsFile("inlier_sweep_results_1img_pair.txt");
    resultsFile << "inliernumformeasure,method,rot_err_deg,trans_err_deg,epi_err_px,inlier_ratio\n";

    // single image pair (1, 2)
    const int i = 1;
    StereoPair pair = loader.loadPair(i, i + 1);

    cv::Mat K = loader.loadIntrinsicCV(i);
    CameraPose pose1 = loader.loadCameraPose(i);
    CameraPose pose2 = loader.loadCameraPose(i + 1);
    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    DTULoader::getRelativePose(pose1, pose2, R_gt, t_gt);
    cv::Mat grayLeft = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);
    SparseKeyPointMatcher matcher(0.75f);
    MatchResult result = matcher.match(grayLeft, grayRight);

    std::vector<cv::Point2f> ptsL, ptsR;
    SparseKeyPointMatcher::extractPoints(result, ptsL, ptsR);

    if (ptsL.size() < 8)
    {
        std::cout << "Not enough points.\n";
        return -1;
    }

    for (int inlierNum : inlierNumValues)
    {
        inliernumformeasure = inlierNum;
        std::cout << "\n############### inliernumformeasure = " << inliernumformeasure << " ###############\n";

        std::vector<bool> CustomInliers, OpenCVInliers, Custommagsacinliers, Customprosacinliers;

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

        auto evaluateMethod = [&](const std::string &name, const Eigen::Matrix3d &F, const std::vector<bool> &inliers)
        {
            double epi_err = Evaluator::evaluateEpipolarError(F, ptsL, ptsR, inliers);
            double ratio = Evaluator::computeInlierRatio(inliers);

            if (GeometryUtils::extractPoseFromFundamental(F, ptsL, ptsR, inliers, K, R_est, t_est))
            {
                double rot_err = 0.0, trans_err = 0.0;
                Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
                std::cout << name << " Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
                std::cout << name << " Epipolar Error: " << epi_err << " px\n";
                std::cout << name << " Inlier Ratio: " << ratio << "%\n";

                resultsFile << inliernumformeasure << "," << name << "," << rot_err << "," << trans_err << ","
                            << epi_err << "," << ratio << "\n";
            }
            else
            {
                std::cout << name << " Failed to recover pose.\n";
                resultsFile << inliernumformeasure << "," << name << ",,,,\n";
            }
        };

        evaluateMethod("OpenCV RANSAC", F_opencv, OpenCVInliers);
        evaluateMethod("Custom RANSAC", F_custom, CustomInliers);
        evaluateMethod("MAGSAC", F_magsac, Custommagsacinliers);
        evaluateMethod("PROSAC", F_prosac, Customprosacinliers);

        resultsFile.flush(); // keep progress on disk in case a later sweep value crashes

        // visualize epipolar matches
        // VisualizationUtils::displayEpipolarMatches("Custom 8-Point", grayLeft, grayRight, ptsL, ptsR, CustomInliers, F_custom);
        // VisualizationUtils::displayEpipolarMatches("OpenCV Baseline", grayLeft, grayRight, ptsL, ptsR, OpenCVInliers, F_opencv);
        // VisualizationUtils::displayEpipolarMatches("MAGSAC", grayLeft, grayRight, ptsL, ptsR, Custommagsacinliers, F_magsac);
        // VisualizationUtils::displayEpipolarMatches("PROSAC", grayLeft, grayRight, ptsL, ptsR, Customprosacinliers, F_prosac);
        // std::cout << "\nExecution complete! Press any key on the image windows to exit.\n";
        // cv::waitKey(0);
    }

    return 0;
}
