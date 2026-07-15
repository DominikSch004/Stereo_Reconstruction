#include <iostream>
#include <DTULoader.hpp>
#include "Pipeline.hpp"
#include "PipelineConfig.hpp"
#include "PlyUtils.hpp"
#include "Evaluator.hpp"
#include "ImgUtils.hpp"
#include <Eigen/Dense>

int main(int argc, char **argv)
{
    // 0. Load pipeline configuration (per-step backend selection)
    const std::string configPath = (argc > 1) ? argv[1] : "../config.yaml";
    PipelineConfig config;
    try
    {
        config = PipelineConfig::load(configPath);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
    config.print();

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

    // 2. Run the pipeline with the configured backends
    PipelineResult res;
    if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, config, poseLeft.t, poseRight.t))
    {
        std::cerr << "ERROR: Pipeline execution failed.\n";
        return 1;
    }

    EvaluatorParams params = {res, R_gt, t_gt, local_gt_cloud};
    EvaluatorRes metrics = Evaluator::evaluateMetrics(params);
    Evaluator::printMetrics(metrics);

    // 3. Export dense local frame 3D point grids to PLY meshes for cloud inspection
    const std::string plyFilename = "pointcloud.ply";
    std::cout << "Saving cloud to: " << plyFilename << "\n";
    PlyUtils::buildAndSavePLY(
        plyFilename,
        res.denseDisparity, res.Q, res.P1r, res.P2r,
        res.camToWorld, res.rectColor, res.minDisp, res.globalConfidence, config.triangulation);

    return 0;
}
