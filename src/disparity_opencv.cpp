#include "StereoPipeline.hpp"
#include "Disparity.hpp"

int main()
{
    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    PipelineResult res;
    if (!runPipeline(leftPath, rightPath, res))
        return -1;

    const int blockSize = 5;

    cv::Mat disparity =
        Disparity::computeOpenCVSGBM(
            res.rectLeft,
            res.rectRight,
            res.minDisp,
            res.numDisp,
            blockSize);

    buildAndSavePLY(
        disparity,
        res,
        res.numDisp,
        "pointcloud_opencv.ply");

    return 0;
}