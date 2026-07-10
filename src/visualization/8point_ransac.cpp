#include <iostream>
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

    // select by image id, default is dataset 1 (scan1) & illumination 3
    int imgL = 1;
    int imgR = 2;
    StereoPair pair = loader.loadPair(imgL, imgR);

    cv::Mat K = loader.loadIntrinsicCV(imgL);
    CameraPose pose1 = loader.loadCameraPose(imgL);
    CameraPose pose2 = loader.loadCameraPose(imgR);
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
        return -1;
    }

    std::vector<bool> CustomInliers, OpenCVInliers, OpenCVMagsacInliers,
        CustomMagsacInspiredInliers, Customprosacinliers;
    
    Eigen::Matrix3d F_custom = FundamentalMatrix::computeFundamental(ptsL, ptsR, CustomInliers, FundamentalMethod::CustomRANSAC, 1.0, 0.99, 1000);
    Eigen::Matrix3d F_opencv = FundamentalMatrix::computeFundamental(ptsL, ptsR, OpenCVInliers, FundamentalMethod::OpenCVRANSAC, 1.0, 0.99, 1000);
    Eigen::Matrix3d F_opencv_magsac = FundamentalMatrix::computeFundamental(
        ptsL, ptsR, OpenCVMagsacInliers, FundamentalMethod::OpenCVMAGSAC,
        1.0, 0.99, 1000);
    Eigen::Matrix3d F_magsac_inspired = FundamentalMatrix::computeFundamental(
        ptsL, ptsR, CustomMagsacInspiredInliers,
        FundamentalMethod::CustomMAGSACInspiredEstimator, 1.0, 0.99, 1000);
    Eigen::Matrix3d F_prosac = FundamentalMatrix::computeFundamental(ptsL, ptsR, Customprosacinliers, FundamentalMethod::CustomPROSAC, 1.0, 0.99, 1000);


    Eigen::Matrix3d R_est;
    Eigen::Vector3d t_est;
    double rot_err = 0.0, trans_err = 0.0;
    double epi_err_custom = Evaluator::evaluateEpipolarError(F_custom, ptsL, ptsR, CustomInliers);
    double custom_ratio = Evaluator::computeInlierRatio(CustomInliers);

    // evaluate custom
    if (GeometryUtils::extractPoseFromFundamental(F_custom, ptsL, ptsR, CustomInliers, K, R_est, t_est))
    {
        Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
        std::cout << "Custom Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
        std::cout << "Custom Epipolar Error: " << epi_err_custom << " px\n";
        std::cout << "Custom Inlier Ratio: " << custom_ratio << "%\n";
    }
    else
    {
        std::cout << "Custom 8-Point Failed to recover pose.\n";
    }

    // evaluate openCV
    double epi_err_cv = Evaluator::evaluateEpipolarError(F_opencv, ptsL, ptsR, OpenCVInliers);
    double cv_ratio = Evaluator::computeInlierRatio(OpenCVInliers);
    if (GeometryUtils::extractPoseFromFundamental(F_opencv, ptsL, ptsR, OpenCVInliers, K, R_est, t_est))
    {
        Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
        std::cout << "OpenCV Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
        std::cout << "OpenCV Epipolar Error: " << epi_err_cv << " px\n";
        std::cout << "OpenCV Inlier Ratio: " << cv_ratio << "%\n";
    }
    else
    {
        std::cout << "OpenCV 8-point failed to recover pose.\n";
    }

    // evaluate OpenCV MAGSAC++
    double epi_err_opencv_magsac = Evaluator::evaluateEpipolarError(
        F_opencv_magsac, ptsL, ptsR, OpenCVMagsacInliers);
    double opencv_magsac_ratio = Evaluator::computeInlierRatio(OpenCVMagsacInliers);
    if (GeometryUtils::extractPoseFromFundamental(
            F_opencv_magsac, ptsL, ptsR, OpenCVMagsacInliers, K, R_est, t_est))
    {   
        Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
        std::cout << "OpenCV MAGSAC++ Geodesic Rotation: " << rot_err
                  << " deg | Translation: " << trans_err << " deg\n";
        std::cout << "OpenCV MAGSAC++ Epipolar Error: " << epi_err_opencv_magsac << " px\n";
        std::cout << "OpenCV MAGSAC++ Inlier Ratio: " << opencv_magsac_ratio << "%\n";
    }
    else
    {
        std::cout << "OpenCV MAGSAC++ failed to recover pose.\n";
    }

    // evaluate custom MAGSAC-inspired estimator
    double epi_err_magsac_inspired = Evaluator::evaluateEpipolarError(
        F_magsac_inspired, ptsL, ptsR, CustomMagsacInspiredInliers);
    double magsac_inspired_ratio = Evaluator::computeInlierRatio(
        CustomMagsacInspiredInliers);
    if (GeometryUtils::extractPoseFromFundamental(
            F_magsac_inspired, ptsL, ptsR, CustomMagsacInspiredInliers,
            K, R_est, t_est))
    {
        Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
        std::cout << "MAGSAC-inspired Geodesic Rotation: " << rot_err
                  << " deg | Translation: " << trans_err << " deg\n";
        std::cout << "MAGSAC-inspired Epipolar Error: " << epi_err_magsac_inspired << " px\n";
        std::cout << "MAGSAC-inspired Inlier Ratio: " << magsac_inspired_ratio << "%\n";
    }
    else
    {
        std::cout << "MAGSAC-inspired estimator failed to recover pose.\n";
    } 
    
    double epi_err_prosac = Evaluator::evaluateEpipolarError(F_prosac, ptsL, ptsR, Customprosacinliers);
    double prosac_ratio = Evaluator::computeInlierRatio(Customprosacinliers);
    if (GeometryUtils::extractPoseFromFundamental(F_prosac, ptsL, ptsR, Customprosacinliers, K, R_est, t_est))
    {
        Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
        std::cout << "PROSAC Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
        std::cout << "PROSAC Epipolar Error: " << epi_err_prosac << " px\n";
        std::cout << "PROSAC Inlier Ratio: " << prosac_ratio << "%\n";
    }
    else
    {
        std::cout << "PROSAC 8-point failed to recover pose.\n";
    }   

    // visualize epipolar matches
    VisualizationUtils::displayEpipolarMatches("Custom 8-Point", grayLeft, grayRight, ptsL, ptsR, CustomInliers, F_custom);
    VisualizationUtils::displayEpipolarMatches("OpenCV Baseline", grayLeft, grayRight, ptsL, ptsR, OpenCVInliers, F_opencv);
    VisualizationUtils::displayEpipolarMatches("OpenCV MAGSAC++", grayLeft, grayRight, ptsL, ptsR, OpenCVMagsacInliers, F_opencv_magsac);
    VisualizationUtils::displayEpipolarMatches("MAGSAC-inspired", grayLeft, grayRight, ptsL, ptsR, CustomMagsacInspiredInliers, F_magsac_inspired);
    VisualizationUtils::displayEpipolarMatches("PROSAC", grayLeft, grayRight, ptsL, ptsR, Customprosacinliers, F_prosac);
    std::cout << "\nExecution complete! Press any key on the image windows to exit.\n";
    cv::waitKey(0);

    return 0;
}
