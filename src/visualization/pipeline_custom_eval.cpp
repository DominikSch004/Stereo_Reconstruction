#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Pipeline.hpp"
#include "PlyUtils.hpp"

int main() {
    // 1. Define image assets from non-rectified DTU dataset paths
    const std::string leftPath  = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png";
    const std::string rightPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png";

    // 2. Execute End-to-End Custom Sequence (not fully implemented yet)
    // TODO: Implement full custom pipeline
    
    std::vector<std::pair<DisparityMethod, std::string>> methods = {
        {DisparityMethod::OpenCVSGBM, "sgbm"},
        {DisparityMethod::SAD,        "sad"},
        {DisparityMethod::SSD,        "ssd"},
        {DisparityMethod::NCC,        "ncc"}
    };
    
    std::cout << "=== Running Multi-Metric Custom Pipeline Sandbox ===\n";

    for (const auto& [method, name] : methods) {
        std::cout << "\n--> Executing Custom Pipeline with metric: " << name << "...\n";
        
        PipelineResult res;
        if (!Pipeline::runPipeline(leftPath, rightPath, res, PipelineMode::Custom, method)) {
            std::cerr << "WARNING: Custom pipeline tracking tripped/unimplemented for metric " << name << "\n";
            continue; 
        }

        // 3. Export dense local frame 3D point grids to PLY meshes for cloud inspection
        std::string plyFilename = "pointcloud_custom_" + name + ".ply";
        std::cout << "Saving cloud to: " << plyFilename << "\n";
        
        PlyUtils::buildAndSavePLY(
            plyFilename,
            res.denseDisparity, res.Q, res.P1r, res.P2r,
            res.camToWorld, res.rectColor, res.minDisp, TriangulationMethod::Manual
        );
    }

    std::cout << "\n=== All Custom PLY generation layers complete! ===\n";

    return 0;
}