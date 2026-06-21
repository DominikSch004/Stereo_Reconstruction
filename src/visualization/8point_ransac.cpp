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
    const std::string leftPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png";
    const std::string rightPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png";

    DTULoader loader("");
    StereoPair pair = loader.loadPair(leftPath, rightPath);
    if (!pair.imageLeft.data || !pair.imageRight.data)
        return -1;

    cv::Mat K = loader.loadIntrinsicCV(leftPath);
    CameraPose pose1 = loader.loadCameraPose(leftPath);
    CameraPose pose2 = loader.loadCameraPose(rightPath);
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

    std::vector<bool> CustomInliers, OpenCVInliers;
    Eigen::Matrix3d F_custom = FundamentalMatrix::computeFundamental(ptsL, ptsR, CustomInliers, FundamentalMethod::CustomRANSAC, 1.0, 0.99, 1000);
    Eigen::Matrix3d F_opencv = FundamentalMatrix::computeFundamental(ptsL, ptsR, OpenCVInliers, FundamentalMethod::OpenCVRANSAC, 1.0, 0.99, 1000);

    Eigen::Matrix3d R_est;
    Eigen::Vector3d t_est;
    double rot_err = 0.0, trans_err = 0.0;
    double epi_err_custom = Evaluator::evaluateEpipolarError(F_custom, ptsL, ptsR, CustomInliers);

    // evaluate custom
    if (GeometryUtils::extractPoseFromFundamental(F_custom, ptsL, ptsR, CustomInliers, K, R_est, t_est))
    {
        Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
        std::cout << "Custom Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
        std::cout << "Custom Epipolar Error: " << epi_err_custom << " px\n";
    }
    else
    {
        std::cout << "Custom 8-Point Failed to recover pose.\n";
    }

    // evaluate openCV
    double epi_err_cv = Evaluator::evaluateEpipolarError(F_opencv, ptsL, ptsR, OpenCVInliers);
    if (GeometryUtils::extractPoseFromFundamental(F_opencv, ptsL, ptsR, OpenCVInliers, K, R_est, t_est))
    {
        Evaluator::evaluatePose(R_est, t_est, R_gt, t_gt, rot_err, trans_err);
        std::cout << "OpenCV Geodesic Rotation: " << rot_err << " deg | Translation: " << trans_err << " deg\n";
        std::cout << "OpenCV Epipolar Error: " << epi_err_cv << " px\n";
    }
    else
    {
        std::cout << "OpenCV 8-point failed to recover pose.\n";
    }

    // visualize epipolar matches
    VisualizationUtils::displayEpipolarMatches("Custom 8-Point", grayLeft, grayRight, ptsL, ptsR, CustomInliers, F_custom);
    VisualizationUtils::displayEpipolarMatches("OpenCV Baseline", grayLeft, grayRight, ptsL, ptsR, OpenCVInliers, F_opencv);

    std::cout << "\nExecution complete! Press any key on the image windows to exit.\n";
    cv::waitKey(0);

    return 0;
}