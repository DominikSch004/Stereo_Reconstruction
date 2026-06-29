#include <iostream>
#include <DTULoader.hpp>
#include "Pipeline.hpp"
#include "PlyUtils.hpp"
#include <Eigen/Dense>

int main()
{
    // 1. Load data
    DTULoader loader("../data/dtu/");

    // select by image id, default is dataset 1 (scan1) & illumination 3
    StereoPair pair = loader.loadPair(1, 2);
    cv::Mat K = loader.loadIntrinsicCV(1);
    CameraPose poseLeft = loader.loadCameraPose(1);
    CameraPose poseRight = loader.loadCameraPose(2);
    // 2. run pipeline, select which one you want to run below.
    bool runOpenCV = true;
    bool runCustom = true;
    PipelineResult res;
    std::string plyFilename;

    if (runOpenCV)
    {
        std::cout << "=== Running OpenCV Pipeline ===\n";

        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, PipelineMode::OpenCV, poseLeft.t, poseRight.t))
        {
            std::cerr << "WARNING: OpenCV pipeline tracking tripped/unimplemented\n";
        }

        // 3. Export dense local frame 3D point grids to PLY meshes for cloud inspection
        plyFilename = "pointcloud_opencv.ply";
        std::cout << "Saving cloud to: " << plyFilename << "\n";
        PlyUtils::buildAndSavePLY(
            plyFilename,
            res.denseDisparity, res.Q, res.P1r, res.P2r,
            res.camToWorld, res.rectColor, res.minDisp, res.globalConfidence, TriangulationMethod::OpenCV);
    }

    if (runCustom)
    {
        std::cout << "=== Running Custom Pipeline ===\n";

        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, PipelineMode::Custom, poseLeft.t, poseRight.t))
        {
            std::cerr << "WARNING: Custom pipeline tracking tripped/unimplemented\n";
        }

        // 3. Export dense local frame 3D point grids to PLY meshes for cloud inspection
        plyFilename = "pointcloud_custom.ply";
        std::cout << "Saving cloud to: " << plyFilename << "\n";
        PlyUtils::buildAndSavePLY(
            plyFilename,
            res.denseDisparity, res.Q, res.P1r, res.P2r,
            res.camToWorld, res.rectColor, res.minDisp, res.globalConfidence, TriangulationMethod::OpenCV);
    }
    return 0;
}