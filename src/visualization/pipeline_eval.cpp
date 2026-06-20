#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Pipeline.hpp"
#include "PlyUtils.hpp"

int main()
{
    // 1. Define image assets from non-rectified DTU dataset paths
    const std::string leftPath  = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png";
    const std::string rightPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png";

    // 2. Execute End-to-End Baseline Sequence (OpenCV Layers)
    PipelineResult baselineRes;
    if (!Pipeline::runPipeline(leftPath, rightPath, baselineRes, PipelineMode::OpenCV)) {
        std::cerr << "OpenCV Baseline Pipeline execution failed.\n";
        return -1;
    }

    // 3. Execute End-to-End Custom Sequence (not fully implemented yet)
    // TODO: Implement full custom pipeline
    PipelineResult customRes;
    if (!Pipeline::runPipeline(leftPath, rightPath, customRes, PipelineMode::Custom)) {
        std::cerr << "Custom Pipeline execution failed.\n";
        return -1;
    }

    // 4. Export dense local frame 3D point grids to PLY meshes for cloud inspection
    std::cout << "\nExporting reconstructed point clouds to disk...\n";
    
    PlyUtils::buildAndSavePLY(
        "pointcloud_baseline.ply",
        baselineRes.denseDisparity, baselineRes.Q, baselineRes.P1r, baselineRes.P2r,
        baselineRes.camToWorld, baselineRes.rectColor, baselineRes.minDisp, TriangulationMethod::OpenCV
    );

    PlyUtils::buildAndSavePLY(
        "pointcloud_custom.ply",
        customRes.denseDisparity, customRes.Q, customRes.P1r, customRes.P2r,
        customRes.camToWorld, customRes.rectColor, customRes.minDisp, TriangulationMethod::Manual
    );

    // 5. Visual Evaluation: Side-by-Side Disparity Map Rendering
    cv::Mat baselineViz, customViz;
    cv::normalize(baselineRes.denseDisparity, baselineViz, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(customRes.denseDisparity,  customViz,  0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat baselineColor, customColor;
    cv::applyColorMap(baselineViz, baselineColor, cv::COLORMAP_JET);
    cv::applyColorMap(customViz,  customColor,  cv::COLORMAP_JET);

    cv::Mat comparison;
    cv::hconcat(baselineColor, customColor, comparison);
    
    cv::imshow("Pipeline Evaluation: OpenCV SGBM Baseline (Left) vs. Custom SAD Engine (Right)", comparison);
    std::cout << "Press any key in the window to exit application loops.\n";
    cv::waitKey(0);

    return 0;
}