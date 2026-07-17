#include "Experiments.hpp"
#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "FundamentalMatrix.hpp"
#include "GeometryUtils.hpp"
#include "Evaluator.hpp"
#include "ImgUtils.hpp"
#include <filesystem>

int main()
{
    // 0. Setup & CSV Initialization
    std::string out_dir = "../experiments";
    if (!std::filesystem::exists(out_dir))
    {
        std::filesystem::create_directories(out_dir);
    }

    // Setup CSV for Experiment 1 (Ratio Test)
    std::string path_ratio = out_dir + "/svd_ratio_benchmark.csv";
    std::ofstream csv_ratio(path_ratio);
    if (!csv_ratio.is_open())
    {
        std::cerr << "Failed to open " << path_ratio << " for writing.\n";
        return -1;
    }
    csv_ratio << "image_pair_start,total_points,outlier_ratio,rot_error_deg,trans_error_deg,epipolar_error\n";

    // Setup CSV for Experiment 2 (Magnitude Test)
    std::string path_magnitude = out_dir + "/svd_magnitude_benchmark.csv";
    std::ofstream csv_magnitude(path_magnitude);
    if (!csv_magnitude.is_open())
    {
        std::cerr << "Failed to open " << path_magnitude << " for writing.\n";
        return -1;
    }
    csv_magnitude << "image_pair_start,total_points,outlier_magnitude,rot_error_deg,trans_error_deg,epipolar_error\n";

    // Setup CSV for Experiment 3 (Inlier Count Test)
    std::string path_inliers = out_dir + "/svd_inliers_benchmark.csv";
    std::ofstream csv_inliers(path_inliers);
    if (!csv_inliers.is_open())
    {
        std::cerr << "Failed to open " << path_inliers << " for writing.\n";
        return -1;
    }
    csv_inliers << "image_pair_start,num_inliers,rot_error_deg,trans_error_deg,epipolar_error\n";

    DTULoader loader("../data/dtu/");
    std::random_device rd;
    std::mt19937 rng(rd());

    int startLeftImageId = 30;
    int datasetId = 6;
    int iterations = 10;
    // random iterations we want to do per experiment
    int randomIterations = 50;

    for (int i = startLeftImageId; i <= startLeftImageId + iterations; i++)
    {
        std::cout << "\n--- Processing Pair " << i << " and " << i + 1 << " ---\n";

        StereoPair pair = loader.loadPair(i, i + 1, datasetId);
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
            std::cout << "Not enough points. Skipping pair.\n";
            continue;
        }

        // *** Evaluate Initial Inliers ***

        std::vector<cv::Point2f> inliersL, inliersR;
        std::vector<bool> inlierMask(ptsL.size(), false);

        // // Commented Out: Load fundamental directly from dataset.
        // cv::Mat F_cv = loader.loadFundamental(i, i + 1);
        // Eigen::Matrix3d F_gt = toEigenMat(F_cv);

        // double threshold = 10.0;
        // double threshold_sq = threshold * threshold;
        // int inlier_count = 0;
        // for (size_t j = 0; j < ptsL.size(); j++)
        // {
        //     double err_sq = FundamentalMatrix::sampsonError(F_gt, ptsL[j], ptsR[j]);
        //     if (err_sq <= threshold_sq)
        //     {
        //         inliersL.push_back(ptsL[j]);
        //         inliersR.push_back(ptsR[j]);
        //         inlierMask[j] = true;
        //     }
        // }

        // Load fundamental from custom MAGSAC
        VisualizationData vis;
        Eigen::Matrix3d F_gt = FundamentalMatrix::computeFundamental(ptsL, ptsR, inlierMask, rng, vis, FundamentalMethod::CustomMAGSAC, 10.0, 0.99, 1000000);

        for (size_t j = 0; j < ptsL.size(); j++)
        {
            if (inlierMask[j])
            {
                inliersL.push_back(ptsL[j]);
                inliersR.push_back(ptsR[j]);
            }
        }
        if (inliersL.size() < 8)
        {
            std::cout << "Not enough inliers. Skipping pair.\n";
            continue;
        }

        cv::Mat R_est;
        cv::Mat t_est;
        cv::Mat F_cv = toCvMat(F_gt);

        if (GeometryUtils::extractPoseFromFundamental(F_gt, ptsL, ptsR, inlierMask, K, R_est, t_est))
        {
            // Evaluation
            EightPointParams initParams = {ptsL, ptsR, F_cv, inlierMask, R_gt, t_gt, toEigenMat(R_est), toEigenVec(t_est)};
            EightPointRes initResult = Evaluator::evaluateEightPoint(initParams);
            std::cout << "[Initial Check] \n";
            Evaluator::printEightPoint(initResult);
        }

        // *** Experiment 1: Ratio Degradation ***
        std::cout << "Running Experiment 1: Ratio Degradation...\n";

        double initRatio = 0.0;
        double finalRatio = 0.9;
        double jump = 0.01;
        bool keepTotalNumberEqual = true;
        // add contamination to inliers progressively
        ContaminatedPointSets contaminatedPoints = Experiments::ratioDegradation(inliersL, inliersR, grayLeft.size(), rng, initRatio,
                                                                                 finalRatio, jump, keepTotalNumberEqual, randomIterations);

        for (size_t step = 0; step < contaminatedPoints.L.size(); step++)
        {
            int step_idx = step / randomIterations;
            double target_ratio = initRatio + (step_idx * jump);
            std::vector<cv::Point2f> mixedL = contaminatedPoints.L[step];
            std::vector<cv::Point2f> mixedR = contaminatedPoints.R[step];
            Eigen::Matrix3d F_est = FundamentalMatrix::compute8Point(mixedL, mixedR);

            cv::Mat R_est_ex1;
            cv::Mat t_est_ex1;
            cv::Mat F_cv_ex1 = toCvMat(F_est);

            if (GeometryUtils::extractPoseFromFundamental(F_est, ptsL, ptsR, inlierMask, K, R_est_ex1, t_est_ex1))
            {
                // Evaluation
                EightPointParams ex1 = {ptsL, ptsR, F_cv_ex1, inlierMask, R_gt, t_gt, toEigenMat(R_est_ex1), toEigenVec(t_est_ex1)};
                EightPointRes resultEx1 = Evaluator::evaluateEightPoint(ex1);

                // Add to csv
                csv_ratio << i << "," << mixedL.size() << ","
                          << target_ratio << "," << resultEx1.rot_error_deg << "," << resultEx1.trans_error_deg << "," << resultEx1.epipolar_error << "\n";
            }
            else
            {
                double epi_err_fallback = Evaluator::evaluateEpipolarError(F_cv_ex1, ptsL, ptsR, inlierMask);
                // pose extraction failed
                csv_ratio << i << "," << mixedL.size() << ","
                          << target_ratio << ",NaN,NaN," << epi_err_fallback << "\n";
            }
        }

        // *** Experiment 2: Magnitude Degradation ***
        std::cout << "Running Experiment 2: Magnitude Degradation...\n";

        double initMag = 1.0;
        double finalMag = 50.0;
        double magJump = 2.0;

        ContaminatedPointSets magnitudePoints = Experiments::magnitudeDegradation(inliersL, inliersR, grayLeft.size(), rng, 0.30,
                                                                                  initMag, finalMag, magJump, randomIterations);

        for (size_t step = 0; step < magnitudePoints.L.size(); step++)
        {
            int step_idx = step / randomIterations;
            double magnitude = initMag + (step_idx * magJump);

            std::vector<cv::Point2f> perturbedL = magnitudePoints.L[step];
            std::vector<cv::Point2f> perturbedR = magnitudePoints.R[step];

            Eigen::Matrix3d F_est = FundamentalMatrix::compute8Point(perturbedL, perturbedR);

            cv::Mat R_est_ex2;
            cv::Mat t_est_ex2;
            cv::Mat F_cv_ex2 = toCvMat(F_est);

            if (GeometryUtils::extractPoseFromFundamental(F_est, ptsL, ptsR, inlierMask, K, R_est_ex2, t_est_ex2))
            {
                // Evaluation
                EightPointParams ex2 = {ptsL, ptsR, F_cv_ex2, inlierMask, R_gt, t_gt, toEigenMat(R_est_ex2), toEigenVec(t_est_ex2)};
                EightPointRes resultEx2 = Evaluator::evaluateEightPoint(ex2);

                csv_magnitude << i << "," << perturbedL.size() << ","
                              << magnitude << "," << resultEx2.rot_error_deg << "," << resultEx2.trans_error_deg << "," << resultEx2.epipolar_error << "\n";
            }
            else
            {
                double epi_err_fallback = Evaluator::evaluateEpipolarError(F_cv_ex2, ptsL, ptsR, inlierMask);
                csv_magnitude << i << "," << perturbedL.size() << ","
                              << magnitude << ",NaN,NaN," << epi_err_fallback << "\n";
            }
        }
        // *** Experiment 3: Progressive Increase of Inlier Sampling ***
        std::cout << "Running Experiment 3: Progressive Increase of Inlier Sampling...\n";

        int initialSamplingCount = 20;
        int skip = 10;
        ContaminatedPointSets subsetPoints = Experiments::inlierSubsets(inliersL, inliersR, rng, randomIterations, initialSamplingCount, -1, skip);

        for (size_t step = 0; step < subsetPoints.L.size(); step++)
        {
            std::vector<cv::Point2f> sampleL = subsetPoints.L[step];
            std::vector<cv::Point2f> sampleR = subsetPoints.R[step];

            int current_N = sampleL.size();

            Eigen::Matrix3d F_est = FundamentalMatrix::compute8Point(sampleL, sampleR);

            cv::Mat R_est_ex3;
            cv::Mat t_est_ex3;
            cv::Mat F_cv_ex3 = toCvMat(F_est);

            if (GeometryUtils::extractPoseFromFundamental(F_est, ptsL, ptsR, inlierMask, K, R_est_ex3, t_est_ex3))
            {
                // Evaluation

                EightPointParams ex3 = {ptsL, ptsR, F_cv_ex3, inlierMask, R_gt, t_gt, toEigenMat(R_est_ex3), toEigenVec(t_est_ex3)};
                EightPointRes resultEx3 = Evaluator::evaluateEightPoint(ex3);

                csv_inliers << i << "," << current_N << ","
                            << resultEx3.rot_error_deg << "," << resultEx3.trans_error_deg << "," << resultEx3.epipolar_error << "\n";
            }
            else
            {
                double epi_err_fallback = Evaluator::evaluateEpipolarError(F_cv_ex3, ptsL, ptsR, inlierMask);
                csv_inliers << i << "," << current_N << ",NaN,NaN," << epi_err_fallback << "\n";
            }
        }
    }

    // Close all CSV files cleanly
    csv_ratio.close();
    csv_magnitude.close();
    csv_inliers.close();
    std::cout << "\nBenchmarking complete. Data written to /experiments in svd_ratio_benchmark.csv, svd_magnitude_benchmark.csv & svd_inliers_benchmark.csv\n";
    return 0;
}