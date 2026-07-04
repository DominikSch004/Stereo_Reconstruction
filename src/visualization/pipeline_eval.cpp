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

    // Load full absolute poses for both cameras
    CameraPose poseLeft = loader.loadCameraPose(viewLeft);
    CameraPose poseRight = loader.loadCameraPose(viewRight);

    cv::Mat K = toCvMat(poseLeft.K);

    // Calculate Ground Truth relative pose
    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    DTULoader::getRelativePose(poseLeft, poseRight, R_gt, t_gt);

    // load point cloud from dataset 1
    std::vector<cv::Point3f> global_gt_cloud = loader.loadPointCloud();
    std::vector<cv::Point3f> local_gt_cloud;
    local_gt_cloud.reserve(global_gt_cloud.size());

    for (const auto &pt : global_gt_cloud)
    {
        Eigen::Vector3d pt_global(pt.x, pt.y, pt.z);
        Eigen::Vector3d pt_local = poseLeft.R * (pt_global - poseLeft.t);
        local_gt_cloud.push_back(cv::Point3f(pt_local(0), pt_local(1), pt_local(2)));
    }

    // run pipeline, select which one you want to run below.
    bool runOpenCV = true;
    bool runCustom = false;
    PipelineResult res;
    std::string plyFilename;

    if (runOpenCV)
    {
        std::cout << "=== Running OpenCV Pipeline ===\n";

        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, PipelineMode::OpenCV, poseLeft.t, poseRight.t))
        {
            std::cerr << "WARNING: OpenCV pipeline tracking tripped/unimplemented\n";
        }
        else
        {
            EvaluatorParams paramsCV = {res, R_gt, t_gt, local_gt_cloud};
            EvaluatorRes metricsCV = Evaluator::evaluateMetrics(paramsCV);
            Evaluator::printMetrics(metricsCV);

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
        else
        {
            EvaluatorParams paramsCustom = {res, R_gt, t_gt, local_gt_cloud};
            EvaluatorRes metricsCustom = Evaluator::evaluateMetrics(paramsCustom);
            Evaluator::printMetrics(metricsCustom);

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