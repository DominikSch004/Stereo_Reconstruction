#include <iostream>
#include <DTULoader.hpp>
#include "Pipeline.hpp"
#include "PlyUtils.hpp"
#include "Evaluator.hpp"
#include "ImgUtils.hpp"
#include <Eigen/Dense>

int main()
{
    // 1. Load data
    DTULoader loader("../data/dtu/");

    int viewLeft = 1;
    int viewRight = 2;

    // select by image id, default is dataset 1 (scan1) & illumination 3
    StereoPair pair = loader.loadPair(viewLeft, viewRight);
    // Load full absolute poses for both cameras (Reads from disk ONCE per camera)
    CameraPose poseLeft = loader.loadCameraPose(viewLeft);
    CameraPose poseRight = loader.loadCameraPose(viewRight);

    // Grab intrinsics directly from the already-loaded pose! No extra disk reads.
    cv::Mat K = toCvMat(poseLeft.K);

    // Calculate Ground Truth relative pose
    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    DTULoader::getRelativePose(poseLeft, poseRight, R_gt, t_gt);

    // 2. run pipeline, select which one you want to run below.
    bool runOpenCV = true;
    bool runCustom = true;
    PipelineResult res;
    std::string plyFilename;

    if (runOpenCV)
    {
        std::cout << "=== Running OpenCV Pipeline ===\n";

        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, PipelineMode::OpenCV))
        {
            std::cerr << "WARNING: OpenCV pipeline tracking tripped/unimplemented\n";
        }
        else
        {
            EvaluatorParams paramsCV = {res, R_gt, t_gt};
            EvaluatorRes metricsCV = Evaluator::evaluateMetrics(paramsCV);
            Evaluator::printMetrics(metricsCV);

            // Export dense local frame 3D point grids to PLY meshes for cloud inspection
            plyFilename = "pointcloud_opencv.ply";
            std::cout << "Saving cloud to: " << plyFilename << "\n";
            PlyUtils::buildAndSavePLY(
                plyFilename,
                res.denseDisparity, res.Q, res.P1r, res.P2r,
                res.camToWorld, res.rectColor, res.minDisp, TriangulationMethod::OpenCV);
        }
    }

    if (runCustom)
    {
        std::cout << "=== Running Custom Pipeline ===\n";

        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, PipelineMode::Custom))
        {
            std::cerr << "WARNING: Custom pipeline tracking tripped/unimplemented\n";
        }
        else
        {
            EvaluatorParams paramsCustom = {res, R_gt, t_gt};
            EvaluatorRes metricsCustom = Evaluator::evaluateMetrics(paramsCustom);
            Evaluator::printMetrics(metricsCustom);

            // Export dense local frame 3D point grids to PLY meshes for cloud inspection
            plyFilename = "pointcloud_custom.ply";
            std::cout << "Saving cloud to: " << plyFilename << "\n";
            PlyUtils::buildAndSavePLY(
                plyFilename,
                res.denseDisparity, res.Q, res.P1r, res.P2r,
                res.camToWorld, res.rectColor, res.minDisp, TriangulationMethod::OpenCV);
        }
    }
    return 0;
}