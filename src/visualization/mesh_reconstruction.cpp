#include <iostream>
#include <DTULoader.hpp>
#include "Pipeline.hpp"
#include "PlyUtils.hpp"
#include "MeshUtils.hpp"
#include <Eigen/Dense>

int main()
{
    // 1. Load data
    DTULoader loader("../data/dtu/");

    StereoPair pair = loader.loadPair(1, 2);
    cv::Mat K = loader.loadIntrinsicCV(1);
    CameraPose poseLeft = loader.loadCameraPose(1);
    CameraPose poseRight = loader.loadCameraPose(2);

    // 2. run pipeline, select which one below.
    bool runOpenCV = true;
    bool runCustom = false;
    PipelineResult res;
    std::string meshFilename;

    // edgeThreshold in world units (mm for DTU). Tune empirically by inspecting output --
    // too tight and continuous surfaces get holes, too loose and foreground/background
    // get bridged across depth discontinuities.
    const float edgeThreshold = 10.0f;

    if (runOpenCV)
    {
        std::cout << "=== Running OpenCV Pipeline ===\n";

        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, PipelineMode::OpenCV, poseLeft.t, poseRight.t))
        {
            std::cerr << "WARNING: OpenCV pipeline tracking tripped/unimplemented\n";
        }

        // 3. Build and export the grid-connectivity mesh for visual inspection
        Mesh mesh = MeshUtils::buildMesh(
            res.denseDisparity, res.Q, res.P1r, res.P2r,
            res.camToWorld, res.rectColor, res.minDisp,
            TriangulationMethod::OpenCV, edgeThreshold);

        meshFilename = "mesh_opencv.ply";
        std::cout << "Saving mesh to: " << meshFilename << "\n";
        MeshUtils::saveMeshPLY(meshFilename, mesh);
    }

    if (runCustom)
    {
        std::cout << "=== Running Custom Pipeline ===\n";

        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, PipelineMode::Custom, poseLeft.t, poseRight.t))
        {
            std::cerr << "WARNING: Custom pipeline tracking tripped/unimplemented\n";
        }

        // 3. Build and export the grid-connectivity mesh for visual inspection
        Mesh mesh = MeshUtils::buildMesh(
            res.denseDisparity, res.Q, res.P1r, res.P2r,
            res.camToWorld, res.rectColor, res.minDisp,
            TriangulationMethod::OpenCV, edgeThreshold);

        meshFilename = "mesh_custom.ply";
        std::cout << "Saving mesh to: " << meshFilename << "\n";
        MeshUtils::saveMeshPLY(meshFilename, mesh);
    }

    return 0;
}