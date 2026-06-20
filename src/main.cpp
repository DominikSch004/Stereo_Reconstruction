#include "Pipeline.hpp"
#include "Disparity.hpp"
#include <iostream>

int main()
{
    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    std::cout << "=== Running Stereo Pipeline Setup ===\n";
    PipelineResult res;
    if (!runPipeline(leftPath, rightPath, res))
    {
        std::cerr << "ERROR: Pipeline execution failed.\n";
        return -1;
    }

    // Block configuration assignments
    const int blockSzCustom = 11;
    const int blockSzSgbm   = 9;
    const int blockSzOpenCV = 5;

    // NCC
    std::cout << "\n--- Computing Custom NCC Disparity ---\n";
    cv::Mat dispNCC = Disparity::computeNCC(res.rectLeft, res.rectRight, res.minDisp, res.numDisp, blockSzCustom);
    buildAndSavePLY(dispNCC, res, res.numDisp, "pointcloud_ncc.ply");

    // SAD
    std::cout << "\n--- Computing Custom SAD Disparity ---\n";
    cv::Mat dispSAD = Disparity::computeSAD(res.rectLeft, res.rectRight, res.minDisp, res.numDisp, blockSzCustom);
    buildAndSavePLY(dispSAD, res, res.numDisp, "pointcloud_sad.ply");

    // SSD
    std::cout << "\n--- Computing Custom SSD Disparity ---\n";
    cv::Mat dispSSD = Disparity::computeSSD(res.rectLeft, res.rectRight, res.minDisp, res.numDisp, blockSzCustom);
    buildAndSavePLY(dispSSD, res, res.numDisp, "pointcloud_ssd.ply");

    // SGBM Custom Configuration
    std::cout << "\n--- Computing Custom SGBM Disparity ---\n";
    cv::Mat dispSGBM = Disparity::computeSGBM(res.rectLeft, res.rectRight, res.minDisp, res.numDisp, blockSzSgbm);
    buildAndSavePLY(dispSGBM, res, res.numDisp, "pointcloud_sgbm.ply");

    // OpenCV Calibration-Driven SGBM
    std::cout << "\n--- Computing OpenCV SGBM Disparity ---\n";
    cv::Mat dispOpenCV = Disparity::computeOpenCVSGBM(res.rectLeft, res.rectRight, res.minDisp, res.numDisp, blockSzOpenCV);
    buildAndSavePLY(dispOpenCV, res, res.numDisp, "pointcloud_opencv.ply");

    std::cout << "\n=== All disparity evaluations completed successfully ===\n";
    return 0;
}