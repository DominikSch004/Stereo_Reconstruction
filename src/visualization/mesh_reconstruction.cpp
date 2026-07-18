#include <iostream>
#include <DTULoader.hpp>
#include "Pipeline.hpp"
#include "PlyUtils.hpp"
#include "MeshUtils.hpp"
#include <Eigen/Dense>

int main(int argc, char **argv)
{
    // 0. Load pipeline configuration
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

    StereoPair pair = loader.loadPair(config.imageLeftId, config.imageRightId, config.datasetId, config.illuminationId);
    cv::Mat K = loader.loadIntrinsicCV(config.imageLeftId);
    CameraPose poseLeft = loader.loadCameraPose(config.imageLeftId);
    CameraPose poseRight = loader.loadCameraPose(config.imageRightId);

    // 2. Run the pipeline
    std::cout << "=== Running Pipeline ===\n";

    PipelineResult res;
    if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, config, poseLeft.t, poseRight.t))
    {
        std::cerr << "ERROR: Pipeline execution failed.\n";
        return -1;
    }

    // edgeThreshold in world units (mm for DTU). Tune empirically by inspecting output --
    // too tight and continuous surfaces get holes, too loose and foreground/background
    // get bridged across depth discontinuities.
    const float edgeThreshold = 10.0f;

    // 3. Build and export the grid-connectivity mesh for visual inspection
    Mesh mesh = MeshUtils::buildMesh(
        res.denseDisparity, res.Q, res.P1r, res.P2r,
        res.camToWorld, res.rectColor, res.minDisp,
        TriangulationMethod::OpenCV, edgeThreshold);

    const std::string meshFilename = (config.disparity == DisparityMethod::OpenCVSGBM) ? "mesh_opencv.ply" : "mesh_custom.ply";
    std::cout << "Saving mesh to: " << meshFilename << "\n";
    MeshUtils::saveMeshPLY(meshFilename, mesh);

    return 0;
}
