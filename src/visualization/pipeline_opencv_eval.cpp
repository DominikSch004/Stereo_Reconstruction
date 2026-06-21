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

    // 2. Execute End-to-End Baseline Sequence (OpenCV Layers) with all the diff disparity methods for evaluation

    // Define all the matching methods we want to sweep through
    std::vector<std::pair<DisparityMethod, std::string>> methods = {
        {DisparityMethod::OpenCVSGBM, "sgbm"},
        {DisparityMethod::SAD,        "sad"},
        {DisparityMethod::SSD,        "ssd"},
        {DisparityMethod::NCC,        "ncc"}
    };

    std::cout << "=== Running Multi-Metric OpenCV Baseline Evaluation ===\n";

    for (const auto& [method, name] : methods) {
        std::cout << "\n--> Executing OpenCV Pipeline with metric: " << name << "...\n";
        
        PipelineResult res;
        if (!Pipeline::runPipeline(leftPath, rightPath, res, PipelineMode::OpenCV, method)) {
            std::cerr << "ERROR: Pipeline failed for metric " << name << "\n";
            continue;
        }

        // 3. Export dense local frame 3D point grids to PLY meshes for cloud inspection
        std::string plyFilename = "pointcloud_opencv_" + name + ".ply";
        std::cout << "Saving cloud to: " << plyFilename << " (" << res.inPtsL.size() << " tracking points)\n";
        
        PlyUtils::buildAndSavePLY(
            plyFilename,
            res.denseDisparity, res.Q, res.P1r, res.P2r,
            res.camToWorld, res.rectColor, res.minDisp, TriangulationMethod::OpenCV
        );
    }

    std::cout << "\n=== All OpenCV PLY generation layers complete! ===\n";

    return 0;
}